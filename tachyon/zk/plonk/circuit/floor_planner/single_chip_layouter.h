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
#include "tachyon/zk/plonk/circuit/floor_planner/simple_table_layouter.h"
#include "tachyon/zk/plonk/circuit/layouter.h"
#include "tachyon/zk/plonk/circuit/region_column.h"

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
    Region(SingleChipLayouter* layouter, size_t region_index)
        : layouter_(layouter), region_index_(region_index) {}

    // zk::Region<F>::Layouter methods
    Error EnableSelector(std::string_view name, const Selector& selector,
                         size_t offset) override {
      return layouter_->assignment_->EnableSelector(
          name, selector, layouter_->regions_[region_index_] + offset);
    }

    void NameColumn(std::string_view name,
                    const AnyColumnKey& column) override {
      return layouter_->assignment_->NameColumn(name, column);
    }

    Error AssignAdvice(std::string_view name, const AdviceColumnKey& column,
                       size_t offset, AssignCallback to, Cell* cell) override {
      Error error = layouter_->assignment_->AssignAdvice(
          name, column, layouter_->regions_[region_index_] + offset,
          std::move(assign));
      if (error != Error::kNone) return error;
      *cell = {region_index_, offset, column};
      return Error::kNone;
    }

    Error AssignAdviceFromConstant(std::string_view name,
                                   const AdviceColumnKey& column, size_t offset,
                                   const math::RationalField<F>& constant,
                                   Cell* cell) override {
      Error error = layouter_->assignment_->AssignAdvice(
          name, column, offset,
          []() { return Value<math::RationalField<F>>::Known(constant); },
          cell);
      if (error != Error::kNone) return error;
      return ConstrainConstant(*cell, constant);
    }

    Error AssignAdviceFromInstance(std::string_view name,
                                   const InstanceColumnKey& instance,
                                   size_t row, const AdviceColumnKey& advice,
                                   size_t offset,
                                   AssignedCell<F>* assigned_cell) {
      Value<F> value;
      Error error =
          layouter_->assignment_->QueryInstance(instance, row, &value);
      if (error != Error::kNone) return error;

      Cell cell;
      error = AssignAdvice(
          name, advice, offset,
          [&value]() { return math::RationalField<F>(value); }, &cell);
      if (error != Error::kNone) return error;

      error = layouter_->assignment_->Copy(
          cell.column(),
          layouter_->regions_[cell.region_index()] + cell.row_offset(),
          instance, row);
      if (error != Error::kNone) return error;

      *assigned_cell = {cell, value};
      return Error::kNone;
    }

    Error AssignFixed(std::string_view name, const FixedColumnKey& column,
                      size_t offset, AssignCallback assign,
                      Cell* cell) override {
      Error error =
          AssignFixed(name, column, layouter_->regions_[region_index_] + offset,
                      std::move(assign));
      if (error != Error::kNone) return error;
      *cell = {region_index_, offset, column};
      return Error::kNone;
    }

    Error ConstrainConstant(const Cell& cell,
                            const math::RationalField<F>& constant) override {
      constants_.emplace_back(constant, cell);
      return Error::kNone;
    }

    Error ConstrainEqual(const Cell& left, const Cell& right) override {
      return layouter_->assignment_->Copy(
          left.column(),
          layouter_->regions_[left.region_index()] + left.row_offset(),
          right.column(),
          layouter_->regions_[right.region_index()] + right.row_offset());
    }

   private:
    SingleChipLayouter* const layouter_;
    const size_t region_index_;
    // Stores the constants to be assigned, and the cells to which they are
    // copied.
    Constants constants_;
  };

  using AssignRegionCallback = typename Layouter<F>::AssignRegionCallback;
  using AssignTableCallback = typename Layouter<F>::AssignTableCallback;

  const Assignment<F>* assignment() const { return assignment_; }
  const std::vector<FixedColumnKey>& constants() const { return constants_; }
  const std::vector<size_t>& regions() const { return regions_; }
  const absl::flat_hash_map<RegionColumn, size_t>& columns() const {
    return columns_;
  }
  const std::vector<TableColumn>& table_columns() const {
    return table_columns_;
  }

  template <typename CircuitTy, typename Config = typename CircuitTy::Config>
  static Error Synthesize(Assignment<F>* assignment, CircuitTy& circuit,
                          Config config,
                          std::vector<FixedColumnKey> constants) {
    SingleChipLayouter layouter(assignment, std::move(constants));
    return circuit.Synthesize(std::move(config));
  }

  // Layouter<F> methods
  Error AssignRegion(std::string_view name,
                     AssignRegionCallback assign) override {
    size_t region_index = regions_.size();

    // Get shape of the region.
    RegionShape<F> shape(region_index);
    {
      // TODO(chokobole): Add event trace using
      // https://github.com/google/perfetto.
      VLOG(1) << "Assign region 1st pass: " << name;
      Region region(&shape);
      Error error = assign.Run(region);
      if (error != Error::kNone) return error;
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
      size_t column_start = columns_[it->first];
      if (column_start != 0 && log_region_info) {
        VLOG(3) << "columns " << column.ToString()
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
    for (auto it = shape.columns().begin(); it != shape.columns().end(); ++it) {
      columns_[column] = region_start + shape.row_count();
    }

    // Assign region cells.
    assignment_->EnterRegion(std::move(name));
    Region region(this, region_index);
    {
      // TODO(chokobole): Add event trace using
      // https://github.com/google/perfetto.
      VLOG(1) << "Assign region 2nd pass: " << name;
      Error error = assign.Run(&region);
      if (error != Error::kNone) return error;
    }
    assignment_->ExitRegion();

    // Assign constants. For the simple floor planner, we assign constants in
    // order in the first `constants` column.
    if (constants_.empty()) {
      if (!region.constants.empty())
        return Error::kNotEnoughColumnsForConstants;
    } else {
      const FixedColumnKey& constants_column = constants_[0];
      size_t& next_constant_row = columns_[RegionColumn(constants_column)];
      for (const Constant& constant : region.constants) {
        const math::RationalField<F>& value = constant.value;
        const Cell& advice = constant.cell;
        Error error = assignment_->AssignFixed(
            [&value]() {
              return absl::Substitute("Constant($0)",
                                      value.Evaluate().ToString());
            },
            constants_column, next_constant_row,
            [&value]() { return Value<F>::Known(value); });
        if (error != Error::kNone) return error;
        error = assignment_->Copy(
            constants_column, next_constant_row, advice.column(),
            regions_[advice.region_index()] + advice.row_offset());
        if (error != Error::kNone) return error;
        ++next_constant_row;
      }
    }
    return Error::kNone;
  }

  Error AssignTable(std::string_view name,
                    AssignTableCallback assign) override {
    // Maintenance hazard: there is near-duplicate code in
    // |v1::AssignmentPass::AssignTable|. Assign table cells.
    assignment_->EnterRegion(name);
    SimpleTableLayouter table_layouter(&assignment_, &table_columns_);
    Error error = std::move(assign).Run(table);
    if (error != Error::kNone) return error;
    const absl::flat_hash_map<TableColumn, SimpleTableLayouter<F>::Value>&
        values = table.values();
    assignment_->ExitRegion();

    // Check that all table columns have the same length |first_unused|,
    // and all cells up to that length are assigned.
    std::optional<size_t> first_unused = 0;
    std::vector<std::optional<size_t>> assigned_sizes = base::Map(
        values.begin(), values.end(),
        [](const SimpleTableLayouter<F>::Value& value) {
          if (std::all_of(value.assigned.begin(), value.assigned.end(),
                          base::identity<bool>())) {
            return std::optional<size_t>(value.assigned.size());
          } else {
            return std::nullopt;
          }
        });
    for (const std::optional<size_t>& assigned_size : assigned_sizes) {
      if (!assigned_size.has_value()) return Error::kSynthesis;

      if (first_unused.has_value()) {
        if (first_unused.value() != assigned_size.value())
          return Error::kSynthesis;
      } else {
        first_unused = assigned_size;
      }
    }

    if (!first_unused.has_value()) return Error::kSynthesis;

    // Record these columns so that we can prevent them from being used again.
    for (auto it = values.begin(); it != values.end(); ++it) {
      table_columns_.push_back(it->first);
      // |it->second.default| must have value because we must have assigned
      // at least one cell in each column, and in that case we checked
      // that all cells up to |first_unused| were assigned.
      Error error = assignment_->FillFromRow(it->first, first_unused.value(),
                                             it->second.default);
      if (error != Error::kNone) return error;
    }

    return Error::kNone;
  }

  Error ConstrainInstance(const Cell& cell, const InstanceColumnKey& column,
                          size_t row) override {
    return assignment_->Copy(cell.column(),
                             regions_[cell.region_index()] + cell.row_offset(),
                             column, row);
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

  SingleChipLayouter(Assignment<F>* assignment,
                     std::vector<FixedColumnKey> constants)
      : assignment_(assignment), constants_(std::move(constants)) {}

  Assignment<F>* const assignment_;
  std::vector<FixedColumnKey> constants_;
  // Stores the starting row for each region.
  std::vector<size_t> regions_;
  // Stores the first empty row for each column.
  absl::flat_hash_map<RegionColumn, size_t> columns_;
  // Stores the table fixed columns.
  std::vector<TableColumn> table_columns_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SINGLE_CHIP_LAYOUTER_H_
