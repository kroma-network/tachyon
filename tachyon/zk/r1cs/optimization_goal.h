// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_ZK_R1CS_OPTIMIZATION_GOAL_H_
#define TACHYON_ZK_R1CS_OPTIMIZATION_GOAL_H_

namespace tachyon::zk::r1cs {

enum class OptimizationGoal {
  // Minimize the number of constraints.
  kConstraints,
  // Minimize the total weight of the constraints (the number of nonzero
  // entries across all constraints).
  kWeight,
};

}  // namespace tachyon::zk::r1cs

#endif  // TACHYON_ZK_R1CS_OPTIMIZATION_GOAL_H_
