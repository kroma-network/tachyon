// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_FLOOR_PLANNER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_FLOOR_PLANNER_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/base/column_key.h"
#include "tachyon/zk/plonk/circuit/assignment.h"

namespace tachyon::zk {

template <typename Circuit>
class FloorPlanner {
 public:
  using F = typename Circuit::Field;
  using Config = typename Circuit::Config;

  virtual void Synthesize(Assignment<F>* assignment, const Circuit& circuit,
                          Config&& config,
                          const std::vector<FixedColumnKey>& constants) = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_FLOOR_PLANNER_H_
