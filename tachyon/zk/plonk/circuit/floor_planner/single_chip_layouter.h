#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SINGLE_CHIP_LAYOUTER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SINGLE_CHIP_LAYOUTER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "tachyon/base/functional/identity.h"
#include "tachyon/base/logging.h"
#include "tachyon/zk/plonk/circuit/floor_planner/simple_lookup_table_layouter.h"
#include "tachyon/zk/plonk/circuit/layouter.h"
#include "tachyon/zk/plonk/circuit/region_column.h"
#include "tachyon/zk/plonk/circuit/region_shape.h"

namespace tachyon::zk {

template <typename F>
class SingleChipLayouter : public Layouter<F> {
 public:
  struct Constant {
    math::RationalField<F> value;
    Cell cell;
  };
  using Constants = std::vector<Constant>;

  class Region : public zk::Region<F>::Layouter {
   public:
    using AssignCallback = typename zk::Region<F>::Layouter::AssignCallback;

    Region(SingleChipLayouter* layouter, size_t region_index)
        : layouter_(layouter), region_index_(region_index) {}

    const Constants& constants() const { return constants_; }

    // zk::Region<F>::Layouter methods
    void EnableSelector(std::string_view name, const Selector& selector,
                        size_t offset) override {
      layouter_->assignment_->EnableSelector(
          name, selector, layouter_->regions_[region_index_] + offset);
    }

    void NameColumn(std::string_view name,
                    const AnyColumnKey& column) override {
      layouter_->assignment_->NameColumn(name, column);
    }

    Cell AssignAdvice(std::string_view name, const AdviceColumnKey& column,
                      size_t offset, AssignCallback assign) override {
      layouter_->assignment_->AssignAdvice(
          name, column, layouter_->regions_[region_index_] + offset,
          std::move(assign));
      return {region_index_, offset, column};
    }

    Cell AssignAdviceFromConstant(
        std::string_view name, const AdviceColumnKey& column, size_t offset,
        const math::RationalField<F>& constant) override {
      Cell cell = AssignAdvice(name, column, offset, [&constant]() {
        return Value<math::RationalField<F>>::Known(constant);
      });
      ConstrainConstant(cell, constant);
      return cell;
    }

    AssignedCell<F> AssignAdviceFromInstance(std::string_view name,
                                             const InstanceColumnKey& instance,
                                             size_t row,
                                             const AdviceColumnKey& advice,
                                             size_t offset) override {
      Value<F> value = layouter_->assignment_->QueryInstance(instance, row);

      Cell cell = AssignAdvice(name, advice, offset, [&value]() {
        return math::RationalField<F>(value);
      });

      layouter_->assignment_->Copy(
          cell.column(),
          layouter_->regions_[cell.region_index()] + cell.row_offset(),
          instance, row);

      return {std::move(cell), std::move(value)};
    }

    Cell AssignFixed(std::string_view name, const FixedColumnKey& column,
                     size_t offset, AssignCallback assign) override {
      layouter_->assignment_->AssignFixed(
          name, column, layouter_->regions_[region_index_] + offset,
          std::move(assign));
      return {region_index_, offset, column};
    }

    void ConstrainConstant(const Cell& cell,
                           const math::RationalField<F>& constant) override {
      constants_.emplace_back(constant, cell);
    }

    void ConstrainEqual(const Cell& left, const Cell& right) override {
      layouter_->assignment_->Copy(
          left.column(),
          layouter_->regions_[left.region_index()] + left.row_offset(),
          right.column(),
          layouter_->regions_[right.region_index()] + right.row_offset());
    }

   private:
    // not owned
    SingleChipLayouter* const layouter_;
    const size_t region_index_;
    // Stores the constants to be assigned, and the cells to which they are
    // copied.
    Constants constants_;
  };

  using AssignRegionCallback = typename Layouter<F>::AssignRegionCallback;
  using AssignLookupTableCallback =
      typename Layouter<F>::AssignLookupTableCallback;

  SingleChipLayouter(Assignment<F>* assignment,
                     const std::vector<FixedColumnKey>& constants)
      : assignment_(assignment), constants_(constants) {}

  const Assignment<F>* assignment() const { return assignment_; }
  const std::vector<FixedColumnKey>& constants() const { return constants_; }
  const std::vector<size_t>& regions() const { return regions_; }
  const absl::flat_hash_map<RegionColumn, size_t>& columns() const {
    return columns_;
  }
  const std::vector<LookupTableColumn>& lookup_table_columns() const {
    return lookup_table_columns_;
  }

  // Layouter<F> methods
  void AssignRegion(std::string_view name,
                    AssignRegionCallback assign) override {
    size_t region_index = regions_.size();

    // Get shape of the region.
    RegionShape<F> shape(region_index);
    {
      // TODO(chokobole): Add event trace using
      // https://github.com/google/perfetto.
      VLOG(1) << "Assign region 1st pass: " << name;
      zk::Region<F> region(&shape);
      assign.Run(region);
    }
    size_t row_count = shape.row_count();
    bool log_region_info = row_count >= 40;
    DLOG_IF(INFO, log_region_info)
        << "Region row count \"" << name << "\": " << row_count;

    // Layout this region. We implement the simplest approach here: position
    // the region starting at the earliest row for which none of the columns are
    // in use.
    size_t region_start = 0;
    for (auto it = shape.columns().begin(); it != shape.columns().end(); ++it) {
      size_t column_start = columns_[*it];
      if (column_start != 0 && log_region_info) {
        VLOG(3) << "columns " << it->ToString()
                << " reused between multi regions. start: " << column_start
                << " region: \"" << name << "\"";
      }
      region_start = std::max(region_start, column_start);
    }
    DLOG_IF(INFO, log_region_info)
        << "region \"" << name << "\", idx: " << regions_.size()
        << " start: " << region_start;
    regions_.push_back(region_start);

    // Update column usage information.
    for (const RegionColumn& column : shape.columns()) {
      columns_[column] = region_start + shape.row_count();
    }

    // Assign region cells.
    assignment_->EnterRegion(std::move(name));
    Region region(this, region_index);
    {
      // TODO(chokobole): Add event trace using
      // https://github.com/google/perfetto.
      VLOG(1) << "Assign region 2nd pass: " << name;
      zk::Region<F> zk_region(&region);
      assign.Run(zk_region);
    }
    assignment_->ExitRegion();

    // Assign constants. For the simple floor planner, we assign constants in
    // order in the first `constants` column.
    if (constants_.empty()) {
      CHECK(region.constants().empty()) << "Not enough columns for constants";
    } else {
      const FixedColumnKey& constants_column = constants_[0];
      size_t& next_constant_row = columns_[RegionColumn(constants_column)];
      for (const Constant& constant : region.constants()) {
        const math::RationalField<F>& value = constant.value;
        const Cell& advice = constant.cell;
        std::string name =
            absl::Substitute("Constant($0)", value.Evaluate().ToString());
        assignment_->AssignFixed(name, constants_column, next_constant_row,
                                 [&value]() { return value; });
        assignment_->Copy(
            constants_column, next_constant_row, advice.column(),
            regions_[advice.region_index()] + advice.row_offset());
        ++next_constant_row;
      }
    }
  }

  void AssignLookupTable(std::string_view name,
                         AssignLookupTableCallback assign) override {
    // Maintenance hazard: there is near-duplicate code in
    // |v1::AssignmentPass::AssignLookupTable|. Assign table cells.
    assignment_->EnterRegion(name);
    SimpleLookupTableLayouter<F> lookup_table_layouter(assignment_,
                                                       &lookup_table_columns_);
    {
      LookupTable<F> table(&lookup_table_layouter);
      std::move(assign).Run(table);
    }
    const absl::flat_hash_map<LookupTableColumn,
                              typename SimpleLookupTableLayouter<F>::Value>&
        values = lookup_table_layouter.values();
    assignment_->ExitRegion();

    // Check that all table columns have the same length |first_unused|,
    // and all cells up to that length are assigned.
    std::optional<size_t> first_unused;
    std::vector<std::optional<size_t>> assigned_sizes = base::Map(
        values.begin(), values.end(),
        [](const std::pair<LookupTableColumn,
                           typename SimpleLookupTableLayouter<F>::Value>&
               entry) {
          const auto& [column, value] = entry;
          if (std::all_of(value.assigned.begin(), value.assigned.end(),
                          base::identity())) {
            return std::optional<size_t>(value.assigned.size());
          } else {
            return std::optional<size_t>();
          }
        });
    for (const std::optional<size_t>& assigned_size : assigned_sizes) {
      CHECK(assigned_size.has_value()) << "length is missing";

      if (first_unused.has_value()) {
        CHECK_EQ(first_unused.value(), assigned_size.value())
            << "all table columns must have the same length";
      } else {
        first_unused = assigned_size;
      }
    }

    CHECK(first_unused.has_value())
        << "length is missing, maybe there are no table columns";

    // Record these columns so that we can prevent them from being used again.
    for (const auto& [column, value] : values) {
      lookup_table_columns_.push_back(column);
      // |value.default_value| must have value because we must have assigned
      // at least one cell in each column, and in that case we checked
      // that all cells up to |first_unused| were assigned.
      assignment_->FillFromRow(column.column(), first_unused.value(),
                               value.default_value.value().value());
    }
  }

  void ConstrainInstance(const Cell& cell, const InstanceColumnKey& column,
                         size_t row) override {
    assignment_->Copy(cell.column(),
                      regions_[cell.region_index()] + cell.row_offset(), column,
                      row);
  }

  Value<F> GetChallenge(const Challenge& challenge) const override {
    return assignment_->GetChallenge(challenge);
  }

  Layouter<F>* GetRoot() override { return this; }

  void PushNamespace(std::string_view name) override {
    assignment_->PushNamespace(name);
  }

  void PopNamespace(const std::optional<std::string>& gadget_name) override {
    assignment_->PopNamespace(gadget_name);
  }

 private:
  friend class Region;

  // not owned
  Assignment<F>* const assignment_;
  std::vector<FixedColumnKey> constants_;
  // Stores the starting row for each region.
  std::vector<size_t> regions_;
  // Stores the first empty row for each column.
  absl::flat_hash_map<RegionColumn, size_t> columns_;
  // Stores the table fixed columns.
  std::vector<LookupTableColumn> lookup_table_columns_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SINGLE_CHIP_LAYOUTER_H_
