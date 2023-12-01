// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_RUNNER_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_RUNNER_H_

#include <functional>
#include <vector>

#include "tachyon/base/parallelize.h"
#include "tachyon/zk/base/prover.h"
#include "tachyon/zk/plonk/circuit/table.h"
#include "tachyon/zk/plonk/permutation/permutation_argument.h"
#include "tachyon/zk/plonk/permutation/permutation_committed.h"
#include "tachyon/zk/plonk/permutation/permutation_evaluated.h"
#include "tachyon/zk/plonk/permutation/permutation_proving_key.h"

namespace tachyon::zk {

template <typename Poly, typename Evals>
class PermutationArgumentRunner {
 public:
  PermutationArgumentRunner() = delete;

  // Returns commitments of Zₚ,ᵢ for chunk index i.
  //
  // See Halo2 book to figure out logic in detail.
  // https://zcash.github.io/halo2/design/proving-system/permutation.html
  template <typename PCSTy, typename F>
  static PermutationCommitted<Poly> CommitArgument(
      Prover<PCSTy>* prover, const PermutationArgument& argument,
      Table<Evals>& table, size_t constraint_system_degree,
      const PermutationProvingKey<PCSTy>& permutation_proving_key,
      const F& beta, const F& gamma);

  template <typename PCSTy, typename F>
  static PermutationEvaluated<Poly> EvaluateCommitted(
      Prover<PCSTy>* prover, PermutationCommitted<Poly>&& committed,
      const F& x);

 private:
  template <typename F>
  static std::function<base::ParallelizeCallback3<F>(size_t)>
  CreateNumeratorCallback(
      const std::vector<Ref<const Evals>>& unpermuted_columns,
      const std::vector<Ref<const Evals>>& value_columns, const F& beta,
      const F& gamma);

  template <typename F>
  static std::function<base::ParallelizeCallback3<F>(size_t)>
  CreateDenominatorCallback(
      const std::vector<Ref<const Evals>>& permuted_columns,
      const std::vector<Ref<const Evals>>& value_columns, const F& beta,
      const F& gamma);
};

}  // namespace tachyon::zk

#include "tachyon/zk/plonk/permutation/permutation_argument_runner_impl.h"

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_ARGUMENT_RUNNER_H_
