// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_ASSIGNMENT_PASS_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_ASSIGNMENT_PASS_H_

#include <stddef.h>

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

#include "tachyon/base/logging.h"
#include "tachyon/zk/plonk/circuit/cell.h"
#include "tachyon/zk/plonk/circuit/column_key.h"
#include "tachyon/zk/plonk/circuit/floor_planner/scoped_region.h"
#include "tachyon/zk/plonk/circuit/floor_planner/simple_lookup_table_layouter.h"
#include "tachyon/zk/plonk/circuit/floor_planner/v1/v1_plan.h"
#include "tachyon/zk/plonk/circuit/floor_planner/v1/v1_region.h"
#include "tachyon/zk/plonk/circuit/layouter.h"

namespace tachyon::zk {

// Assigns the circuit.
template <typename F>
class AssignmentPass {
 public:
  using AssignRegionCallback = typename Layouter<F>::AssignRegionCallback;
  using AssignLookupTableCallback =
      typename Layouter<F>::AssignLookupTableCallback;

  explicit AssignmentPass(V1Plan<F>* plan) : plan_(plan) {}

  V1Plan<F>* plan() const { return plan_; }

  void AssignRegion(std::string_view name, AssignRegionCallback assign) {
    ScopedRegion<F> scoped_region(plan_->assignment(), name);
    V1Region<F> v1_region(plan_, region_index_++);
    zk::Region<F> region(&v1_region);
    assign.Run(region);
  }

  void AssignLookupTable(std::string_view name,
                         AssignLookupTableCallback assign) {
    // Maintenance hazard: there is near-duplicate code in
    // |SingleChipLayouter::AssignLookUpTable|.

    // Assign table cells.
    SimpleLookupTableLayouter<F> lookup_table_layouter(plan_->assignment(),
                                                       &plan_->table_columns());
    {
      ScopedRegion<F> scoped_region(plan_->assignment(), name);
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
      plan_->table_columns().push_back(column);
      // |value.default_value| must have value because we must have assigned
      // at least one cell in each column, and in that case we checked
      // that all cells up to |first_unused| were assigned.
      plan_->assignment()->FillFromRow(column.column(), first_unused.value(),
                                       value.default_value.value().value());
    }
  }

  void ConstrainInstance(const Cell& cell, const InstanceColumnKey& instance,
                         size_t row) {
    plan_->assignment()->Copy(
        cell.column(),
        plan_->regions()[cell.region_index()] + cell.row_offset(), instance,
        row);
  }

 private:
  // not owned
  V1Plan<F>* const plan_;
  // Counter tracking which region we need to assign next.
  size_t region_index_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_V1_ASSIGNMENT_PASS_H_
