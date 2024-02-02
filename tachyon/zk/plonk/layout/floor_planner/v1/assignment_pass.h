// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_ASSIGNMENT_PASS_H_
#define TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_ASSIGNMENT_PASS_H_

#include <stddef.h>

#include <string_view>
#include <utility>

#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/layout/cell.h"
#include "tachyon/zk/plonk/layout/floor_planner/lookup_table_assigner.h"
#include "tachyon/zk/plonk/layout/floor_planner/plan_region.h"
#include "tachyon/zk/plonk/layout/floor_planner/scoped_region.h"
#include "tachyon/zk/plonk/layout/floor_planner/v1/v1_plan.h"
#include "tachyon/zk/plonk/layout/layouter.h"

namespace tachyon::zk::plonk {

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
    PlanRegion<F> plan_region(plan_->assignment(), plan_->regions(),
                              region_index_++, plan_->constants());
    Region<F> region(&plan_region);
    assign.Run(region);
  }

  void AssignLookupTable(std::string_view name,
                         AssignLookupTableCallback assign) {
    LookupTableAssigner<F> lookup_table_assigner(plan_->assignment(),
                                                 plan_->table_columns());
    lookup_table_assigner.AssignLookupTable(name, std::move(assign));
  }

  void ConstrainInstance(const Cell& cell, const InstanceColumnKey& instance,
                         RowIndex row) {
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

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_ASSIGNMENT_PASS_H_
