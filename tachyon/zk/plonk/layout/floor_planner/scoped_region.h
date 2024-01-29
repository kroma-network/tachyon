// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_SCOPED_REGION_H_
#define TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_SCOPED_REGION_H_

#include <string_view>

#include "tachyon/zk/plonk/layout/assignment.h"

namespace tachyon::zk {

template <typename F>
struct ScopedRegion {
  ScopedRegion(Assignment<F>* assignment, std::string_view name)
      : assignment_(assignment) {
    assignment_->EnterRegion(name);
  }
  ~ScopedRegion() { assignment_->ExitRegion(); }

  // not owned
  Assignment<F>* const assignment_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LAYOUT_FLOOR_PLANNER_SCOPED_REGION_H_
