#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SINGLE_CHIP_LAYOUTER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_SINGLE_CHIP_LAYOUTER_H_

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "tachyon/base/functional/identity.h"
#include "tachyon/base/logging.h"
#include "tachyon/zk/plonk/circuit/floor_planner/constant.h"
#include "tachyon/zk/plonk/circuit/floor_planner/plan_region.h"
#include "tachyon/zk/plonk/circuit/floor_planner/scoped_region.h"
#include "tachyon/zk/plonk/circuit/floor_planner/simple_lookup_table_layouter.h"
#include "tachyon/zk/plonk/circuit/layouter.h"
#include "tachyon/zk/plonk/circuit/region_column.h"
#include "tachyon/zk/plonk/circuit/region_shape.h"

namespace tachyon::zk {

template <typename F>
class SingleChipLayouter : public Layouter<F> {
 public:
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
      Region<F> region(&shape);
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

    Constants<F> constants;

    // Assign region cells.
    PlanRegion plan_region(assignment_, regions_, region_index, constants);
    {
      ScopedRegion<F> scoped_region(assignment_, name);
      // TODO(chokobole): Add event trace using
      // https://github.com/google/perfetto.
      VLOG(1) << "Assign region 2nd pass: " << name;
      Region<F> region(&plan_region);
      assign.Run(region);
    }

    // Assign constants. For the simple floor planner, we assign constants in
    // order in the first |constants| column.
    if (constants_.empty()) {
      CHECK(plan_region.constants().empty())
          << "Not enough columns for constants";
    } else {
      const FixedColumnKey& constants_column = constants_[0];
      size_t& next_constant_row = columns_[RegionColumn(constants_column)];
      for (const Constant<F>& constant : plan_region.constants()) {
        const math::RationalField<F>& value = constant.value;
        const Cell& advice = constant.cell;
        std::string name =
            absl::Substitute("Constant($0)", value.Evaluate().ToString());
        assignment_->AssignFixed(
            name, constants_column, next_constant_row,
            [&value]() { return Value<math::RationalField<F>>::Known(value); });
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
    // |AssignmentPass::AssignLookupTable|.

    // Assign table cells.
    SimpleLookupTableLayouter<F> lookup_table_layouter(assignment_,
                                                       &lookup_table_columns_);
    {
      ScopedRegion<F> scoped_region(assignment_, name);
      LookupTable<F> table(&lookup_table_layouter);
      std::move(assign).Run(table);
    }
    const absl::flat_hash_map<LookupTableColumn,
                              typename SimpleLookupTableLayouter<F>::Value>&
        values = lookup_table_layouter.values();

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
