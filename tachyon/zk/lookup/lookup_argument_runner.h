// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_H_
#define TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_H_

#include <vector>

#include "gtest/gtest_prod.h"

#include "tachyon/zk/base/entities/prover.h"
#include "tachyon/zk/base/prover_query.h"
#include "tachyon/zk/expressions/evaluator/simple_evaluator.h"
#include "tachyon/zk/lookup/lookup_argument.h"
#include "tachyon/zk/lookup/lookup_committed.h"
#include "tachyon/zk/lookup/lookup_evaluated.h"
#include "tachyon/zk/lookup/lookup_permuted.h"

namespace tachyon::zk {

template <typename Poly, typename Evals>
class LookupArgumentRunner {
 public:
  LookupArgumentRunner() = delete;

  template <typename PCSTy, typename F>
  static LookupPermuted<Poly, Evals> PermuteArgument(
      Prover<PCSTy>* prover, const LookupArgument<F>& argument, const F& theta,
      const SimpleEvaluator<Evals>& evaluator_tpl);

  template <typename PCSTy, typename F>
  static std::vector<LookupPermuted<Poly, Evals>> BatchPermuteArgument(
      Prover<PCSTy>* prover, const std::vector<LookupArgument<F>>& arguments,
      const F& theta, const SimpleEvaluator<Evals>& evaluator_tpl);

  template <typename PCSTy, typename F>
  static LookupCommitted<Poly> CommitPermuted(
      Prover<PCSTy>* prover, LookupPermuted<Poly, Evals>&& permuted,
      const F& beta, const F& gamma);

  template <typename PCSTy, typename F>
  static std::vector<LookupCommitted<Poly>> BatchCommitPermuted(
      Prover<PCSTy>* prover,
      std::vector<LookupPermuted<Poly, Evals>>&& permuteds, const F& beta,
      const F& gamma);

  template <typename PCSTy, typename F>
  static LookupEvaluated<Poly> EvaluateCommitted(
      Prover<PCSTy>* prover, LookupCommitted<Poly>&& committed, const F& x);

  template <typename PCSTy, typename F>
  static std::vector<LookupEvaluated<Poly>> BatchEvaluateCommitted(
      Prover<PCSTy>* prover, std::vector<LookupCommitted<Poly>>&& committeds,
      const F& x);

  template <typename PCSTy, typename F>
  static std::vector<ProverQuery<PCSTy>> OpenEvaluated(
      const Prover<PCSTy>* prover, const LookupEvaluated<Poly>& evaluated,
      const F& x);

  template <typename PCSTy, typename F>
  static std::vector<ProverQuery<PCSTy>> BatchOpenEvaluated(
      const Prover<PCSTy>* prover,
      const std::vector<LookupEvaluated<Poly>>& evaluateds, const F& x);

 private:
  FRIEND_TEST(LookupArgumentRunnerTest, ComputePermutationProduct);

  template <typename F>
  static base::ParallelizeCallback3<F> CreateNumeratorCallback(
      const LookupPermuted<Poly, Evals>& permuted, const F& beta,
      const F& gamma);

  template <typename F>
  static base::ParallelizeCallback3<F> CreateDenominatorCallback(
      const LookupPermuted<Poly, Evals>& permuted, const F& beta,
      const F& gamma);
};

}  // namespace tachyon::zk

#include "tachyon/zk/lookup/lookup_argument_runner_impl.h"

#endif  // TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_H_
