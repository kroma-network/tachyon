// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_FLOOR_PLANNER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_FLOOR_PLANNER_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/circuit/assignment.h"
#include "tachyon/zk/plonk/circuit/column_key.h"

namespace tachyon::zk {

template <typename CircuitTy>
class FloorPlanner {
 public:
  using F = typename CircuitTy::Field;
  using Config = typename CircuitTy::Config;

  virtual void Synthesize(Assignment<F>* assignment, CircuitTy& circuit,
                          Config&& config,
                          const std::vector<FixedColumnKey>& constants) = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_FLOOR_PLANNER_FLOOR_PLANNER_H_
