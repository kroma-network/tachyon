// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_IMPL_H_
#define TACHYON_ZK_PLONK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_IMPL_H_

#include <utility>

#include "tachyon/zk/plonk/lookup/compress_expression.h"
#include "tachyon/zk/plonk/lookup/lookup_argument_runner.h"

namespace tachyon::zk {

template <typename Poly, typename Evals>
template <typename PCSTy, typename ExtendedDomain, typename F>
LookupPermuted<Poly, Evals> LookupArgumentRunner<Poly, Evals>::PermuteArgument(
    Prover<PCSTy, ExtendedDomain>* prover, const LookupArgument<F>& argument,
    const F& theta, const SimpleEvaluator<Evals>& evaluator_tpl) {
  // A_compressed(X) = θᵐ⁻¹A₀(X) + θᵐ⁻²A₁(X) + ... + θAₘ₋₂(X) + Aₘ₋₁(X)
  Evals compressed_input_expression;
  CHECK(CompressExpressions(argument.input_expressions(),
                            prover->domain()->size(), theta, evaluator_tpl,
                            &compressed_input_expression));

  // S_compressed(X) = θᵐ⁻¹S₀(X) + θᵐ⁻²S₁(X) + ... + θSₘ₋₂(X) + Sₘ₋₁(X)
  Evals compressed_table_expression;
  CHECK(CompressExpressions(argument.table_expressions(),
                            prover->domain()->size(), theta, evaluator_tpl,
                            &compressed_table_expression));

  // Permute compressed (InputExpression, TableExpression) pair.
  EvalsPair<Evals> compressed_evals_pair(
      std::move(compressed_input_expression),
      std::move(compressed_table_expression));

  // A'(X), S'(X)
  EvalsPair<Evals> permuted_evals_pair;
  Error err = PermuteExpressionPair(prover, compressed_evals_pair,
                                    &permuted_evals_pair);
  CHECK_EQ(err, Error::kNone);

  // Commit(A'(X))
  BlindedPolynomial<Poly> permuted_input_poly;
  CHECK(prover->CommitEvalsWithBlind(permuted_evals_pair.input(),
                                     &permuted_input_poly));

  // Commit(S'(X))
  BlindedPolynomial<Poly> permuted_table_poly;
  CHECK(prover->CommitEvalsWithBlind(permuted_evals_pair.table(),
                                     &permuted_table_poly));

  return {std::move(compressed_evals_pair), std::move(permuted_evals_pair),
          std::move(permuted_input_poly), std::move(permuted_table_poly)};
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_IMPL_H_
