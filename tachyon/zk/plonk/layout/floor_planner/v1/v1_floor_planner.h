// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_V1_FLOOR_PLANNER_H_
#define TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_V1_FLOOR_PLANNER_H_

#include <stddef.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/zk/plonk/layout/floor_planner/floor_planner.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/measurement_pass.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_pass.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_plan.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_strategy.h"

namespace tachyon::zk::plonk {

// The version 1 |FloorPlanner| provided by "halo2".
//
// - No column optimizations are performed. Circuit configuration is left
//   entirely to the circuit designer.
// - A dual-pass layouter is used to measure regions prior to assignment.
// - Regions are measured as rectangles, bounded on the cells they assign.
// - Regions are laid out using a greedy first-fit strategy, after sorting
//   regions by their "advice area" (number of advice columns * rows).
template <typename Circuit>
class V1FloorPlanner : public FloorPlanner<Circuit> {
 public:
  using F = typename Circuit::Field;
  using Config = typename Circuit::Config;

  void Synthesize(Assignment<F>* assignment, const Circuit& circuit,
                  Config&& config,
                  const std::vector<FixedColumnKey>& constants) override {
    // First pass: measure the regions within the circuit.
    MeasurementPass<F> measure;
    {
      V1Pass<F> pass(&measure);
      circuit.WithoutWitness()->Synthesize(config.Clone(), &pass);
    }
    for (const typename MeasurementPass<F>::Region& region :
         measure.regions()) {
      // TODO(TomTaehoonKim): Add event trace using
      // https://github.com/google/perfetto.
      VLOG(1) << "Region height " << region.name << ": "
              << region.shape.row_count();
    }

    // Planning:
    // - Position the regions.
    V1Plan<F> plan(assignment);
    std::vector<RegionShape<F>> region_shapes =
        base::Map(measure.regions(),
                  [](const typename MeasurementPass<F>::Region& region) {
                    return region.shape;
                  });
    SlotInBiggestAdviceFirstResult result =
        SlotInBiggestAdviceFirst(region_shapes);
    plan.set_regions(std::move(result.region_starts));

    // - Determine how many rows our planned circuit will require.
    RowIndex first_unassigned_row = 0;
    if (!result.column_allocations.empty()) {
      std::vector<RowIndex> unbounded_interval_starts = base::Map(
          result.column_allocations,
          [](const std::pair<RegionColumn, Allocations>& column_allocation) {
            return column_allocation.second.UnboundedIntervalStart();
          });
      first_unassigned_row = *std::max_element(
          unbounded_interval_starts.begin(), unbounded_interval_starts.end());
    }

    // - Position the constants within those rows.
    std::vector<FixedAllocation> fixed_allocations =
        base::Map(constants, [&result](const FixedColumnKey& c) {
          return FixedAllocation(
              c, result.column_allocations[RegionColumn(AnyColumnKey(c))]);
        });

    std::vector<ConstantPosition> constant_positions = base::FlatMap(
        fixed_allocations,
        [first_unassigned_row](const FixedAllocation& fixed_allocation) {
          return base::FlatMap(
              fixed_allocation.rows.FreeIntervals(0, first_unassigned_row),
              [&column = fixed_allocation.column](const EmptySpace& space) {
                std::optional<base::Range<RowIndex>> range = space.Range();
                return range.has_value()
                           ? base::Map(range.value(),
                                       [&column](RowIndex row) {
                                         return ConstantPosition(column, row);
                                       })
                           : std::vector<ConstantPosition>();
              });
        });

    // Second pass:
    // - Assign the regions.
    AssignmentPass<F> assign(&plan);
    {
      V1Pass<F> pass(&assign);
      circuit.Synthesize(std::move(config), &pass);
    }

    // - Assign the constants.
    CHECK_GE(constant_positions.size(), plan.constants().size())
        << "Not enough columns for constants";
    // TODO(TomTaehoonKim): Refac using |base::Zipped| when |ZippedIterator| is
    // fixed to stop producing values once the shorter iterator is exhausted.
    for (size_t i = 0;
         i < std::min(constant_positions.size(), plan.constants().size());
         ++i) {
      plan.assignment()->AssignFixed(
          absl::Substitute("Constant($0)",
                           plan.constants()[i].value.Evaluate().ToString()),
          constant_positions[i].column, constant_positions[i].row,
          [&value = plan.constants()[i].value]() {
            return Value<math::RationalField<F>>::Known(value);
          });
      plan.assignment()->Copy(
          constant_positions[i].column, constant_positions[i].row,
          plan.constants()[i].cell.column(),
          plan.regions()[plan.constants()[i].cell.region_index()] +
              plan.constants()[i].cell.row_offset());
    }
  }

 private:
  struct FixedAllocation {
    FixedAllocation(const FixedColumnKey& column, const Allocations& rows)
        : column(column), rows(rows) {}

    FixedColumnKey column;
    Allocations rows;
  };

  struct ConstantPosition {
    ConstantPosition(const FixedColumnKey& column, RowIndex row)
        : column(column), row(row) {}

    FixedColumnKey column;
    RowIndex row;
  };
};

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_V1_FLOOR_PLANNER_H_
