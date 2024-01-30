// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_V1_PLAN_H_
#define TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_V1_PLAN_H_

#include <utility>
#include <vector>

#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/plonk/layout/assignment.h"
#include "tachyon/zk/plonk/layout/cell.h"
#include "tachyon/zk/plonk/layout/floor_planner/constant.h"
#include "tachyon/zk/plonk/layout/lookup_table_column.h"

namespace tachyon::zk {

template <typename F>
class V1Plan {
 public:
  explicit V1Plan(Assignment<F>* assignment) : assignment_(assignment) {}

  Assignment<F>* assignment() { return assignment_; }
  const std::vector<RowIndex>& regions() const { return regions_; }
  void set_regions(std::vector<RowIndex>&& regions) {
    regions_ = std::move(regions);
  }
  Constants<F>& constants() { return constants_; }
  std::vector<LookupTableColumn>& table_columns() { return table_columns_; }

 private:
  // not owned
  Assignment<F>* const assignment_;
  // Stores the starting row for each region.
  std::vector<RowIndex> regions_;
  // Stores the constants to be assigned, and the cells to which
  // they are copied.
  Constants<F> constants_;
  // Stores the table fixed columns.
  std::vector<LookupTableColumn> table_columns_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_V1_V1_PLAN_H_
