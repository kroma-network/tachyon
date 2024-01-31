// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_H_
#define TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_H_

#include <vector>

#include "gtest/gtest_prod.h"

#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/base/point_set.h"
#include "tachyon/zk/expressions/evaluator/simple_evaluator.h"
#include "tachyon/zk/lookup/lookup_argument.h"
#include "tachyon/zk/lookup/lookup_committed.h"
#include "tachyon/zk/lookup/lookup_evaluated.h"
#include "tachyon/zk/lookup/lookup_permuted.h"

namespace tachyon::zk {

template <typename Poly, typename Evals>
class LookupArgumentRunner {
 public:
  using F = typename Poly::Field;

  LookupArgumentRunner() = delete;

  template <typename PCS>
  static LookupPermuted<Poly, Evals> PermuteArgument(
      ProverBase<PCS>* prover, const LookupArgument<F>& argument,
      const F& theta, const SimpleEvaluator<Evals>& evaluator_tpl);

  template <typename PCS>
  static LookupCommitted<Poly> CommitPermuted(
      ProverBase<PCS>* prover, LookupPermuted<Poly, Evals>&& permuted,
      const F& beta, const F& gamma);

  template <typename PCS>
  static LookupEvaluated<Poly> EvaluateCommitted(
      ProverBase<PCS>* prover, LookupCommitted<Poly>&& committed, const F& x);

  template <typename PCS>
  static std::vector<crypto::PolynomialOpening<Poly>> OpenEvaluated(
      const ProverBase<PCS>* prover, const LookupEvaluated<Poly>& evaluated,
      const F& x, PointSet<F>& points);

 private:
  FRIEND_TEST(LookupArgumentRunnerTest, ComputePermutationProduct);

  static base::ParallelizeCallback3<F> CreateNumeratorCallback(
      const LookupPermuted<Poly, Evals>& permuted, const F& beta,
      const F& gamma);

  static base::ParallelizeCallback3<F> CreateDenominatorCallback(
      const LookupPermuted<Poly, Evals>& permuted, const F& beta,
      const F& gamma);
};

}  // namespace tachyon::zk

#include "tachyon/zk/lookup/lookup_argument_runner_impl.h"

#endif  // TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_H_
