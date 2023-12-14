// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_PINNED_CONSTRAINT_SYSTEM_H_
#define TACHYON_ZK_PLONK_HALO2_PINNED_CONSTRAINT_SYSTEM_H_

#include <optional>
#include <string>
#include <vector>

#include "tachyon/zk/lookup/lookup_argument_stringifier.h"
#include "tachyon/zk/plonk/circuit/phase_stringifier.h"
#include "tachyon/zk/plonk/circuit/query_stringifier.h"
#include "tachyon/zk/plonk/constraint_system.h"
#include "tachyon/zk/plonk/halo2/pinned_gates.h"
#include "tachyon/zk/plonk/permutation/permutation_argument_stringifier.h"

namespace tachyon {
namespace zk::halo2 {

template <typename F>
class PinnedConstraintSystem {
 public:
  explicit PinnedConstraintSystem(const ConstraintSystem<F>& constraint_system)
      : num_fixed_columns_(constraint_system.num_fixed_columns()),
        num_advice_columns_(constraint_system.num_advice_columns()),
        num_instance_columns_(constraint_system.num_instance_columns()),
        num_selectors_(constraint_system.num_selectors()),
        num_challenges_(constraint_system.num_challenges()),
        advice_column_phases_(constraint_system.advice_column_phases()),
        challenge_phases_(constraint_system.challenge_phases()),
        gates_(constraint_system.gates()),
        advice_queries_(constraint_system.advice_queries()),
        instance_queries_(constraint_system.instance_queries()),
        fixed_queries_(constraint_system.fixed_queries()),
        permutation_(constraint_system.permutation()),
        lookups_(constraint_system.lookups()),
        constants_(constraint_system.constants()),
        minimum_degree_(constraint_system.minimum_degree()) {}

  size_t num_fixed_columns() const { return num_fixed_columns_; }
  size_t num_advice_columns() const { return num_advice_columns_; }
  size_t num_instance_columns() const { return num_instance_columns_; }
  size_t num_selectors() const { return num_selectors_; }
  size_t num_challenges() const { return num_challenges_; }
  const std::vector<Phase>& advice_column_phases() const {
    return advice_column_phases_;
  }
  const std::vector<Phase>& challenge_phases() const {
    return challenge_phases_;
  }
  const PinnedGates<F>& gates() const { return gates_; }
  const std::vector<AdviceQueryData>& advice_queries() const {
    return advice_queries_;
  }
  const std::vector<InstanceQueryData>& instance_queries() const {
    return instance_queries_;
  }
  const std::vector<FixedQueryData>& fixed_queries() const {
    return fixed_queries_;
  }
  const PermutationArgument& permutation() const { return permutation_; }
  const std::vector<LookupArgument<F>>& lookups() const { return lookups_; }
  const std::vector<FixedColumnKey>& constants() const { return constants_; }
  const std::optional<size_t>& minimum_degree() const {
    return minimum_degree_;
  }

 private:
  size_t num_fixed_columns_;
  size_t num_advice_columns_;
  size_t num_instance_columns_;
  size_t num_selectors_;
  size_t num_challenges_;
  const std::vector<Phase>& advice_column_phases_;
  const std::vector<Phase>& challenge_phases_;
  PinnedGates<F> gates_;
  const std::vector<AdviceQueryData>& advice_queries_;
  const std::vector<InstanceQueryData>& instance_queries_;
  const std::vector<FixedQueryData>& fixed_queries_;
  PermutationArgument permutation_;
  const std::vector<LookupArgument<F>>& lookups_;
  const std::vector<FixedColumnKey>& constants_;
  const std::optional<size_t>& minimum_degree_;
};

}  // namespace zk::halo2

namespace base::internal {

template <typename F>
class RustDebugStringifier<zk::halo2::PinnedConstraintSystem<F>> {
 public:
  static std::ostream& AppendToStream(
      std::ostream& os, RustFormatter& fmt,
      const zk::halo2::PinnedConstraintSystem<F>& constraint_system) {
    DebugStruct debug_struct = fmt.DebugStruct("PinnedConstraintSystem");
    debug_struct
        .Field("num_fixed_columns", constraint_system.num_fixed_columns())
        .Field("num_advice_columns", constraint_system.num_advice_columns())
        .Field("num_instance_columns", constraint_system.num_instance_columns())
        .Field("num_selectors", constraint_system.num_selectors());
    if (constraint_system.num_challenges() > 0) {
      debug_struct.Field("num_challenges", constraint_system.num_challenges())
          .Field("advice_column_phases",
                 constraint_system.advice_column_phases())
          .Field("challenge_phases", constraint_system.challenge_phases());
    }
    debug_struct.Field("gates", constraint_system.gates())
        .Field("advice_queries", constraint_system.advice_queries())
        .Field("instance_queries", constraint_system.instance_queries())
        .Field("fixed_queries", constraint_system.fixed_queries())
        .Field("permutation", constraint_system.permutation())
        .Field("lookups", constraint_system.lookups())
        .Field("constants", constraint_system.constants())
        .Field("minimum_degree", constraint_system.minimum_degree());
    return os << debug_struct.Finish();
  }
};

}  // namespace base::internal
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_PINNED_CONSTRAINT_SYSTEM_H_
