#ifndef TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_SINGLE_CHIP_LAYOUTER_H_
#define TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_SINGLE_CHIP_LAYOUTER_H_

#include <stddef.h>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "tachyon/base/logging.h"
#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/layout/floor_planner/constant.h"
#include "tachyon/zk/plonk/layout/floor_planner/lookup_table_assigner.h"
#include "tachyon/zk/plonk/layout/floor_planner/plan_region.h"
#include "tachyon/zk/plonk/layout/floor_planner/scoped_region.h"
#include "tachyon/zk/plonk/layout/layouter.h"
#include "tachyon/zk/plonk/layout/region_column.h"
#include "tachyon/zk/plonk/layout/region_shape.h"

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
  const std::vector<RowIndex>& regions() const { return regions_; }
  const absl::flat_hash_map<RegionColumn, RowIndex>& columns() const {
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
    RowIndex row_count = shape.row_count();
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
      RowIndex& next_constant_row = columns_[RegionColumn(constants_column)];
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
    LookupTableAssigner<F> lookup_table_assigner(assignment_,
                                                 lookup_table_columns_);
    lookup_table_assigner.AssignLookupTable(name, std::move(assign));
  }

  void ConstrainInstance(const Cell& cell, const InstanceColumnKey& column,
                         RowIndex row) override {
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
  std::vector<RowIndex> regions_;
  // Stores the first empty row for each column.
  absl::flat_hash_map<RegionColumn, RowIndex> columns_;
  // Stores the table fixed columns.
  std::vector<LookupTableColumn> lookup_table_columns_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_SINGLE_CHIP_LAYOUTER_H_
