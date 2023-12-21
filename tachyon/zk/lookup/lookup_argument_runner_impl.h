// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_IMPL_H_
#define TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_IMPL_H_

#include <utility>
#include <vector>

#include "tachyon/zk/lookup/compress_expression.h"
#include "tachyon/zk/lookup/lookup_argument_runner.h"
#include "tachyon/zk/plonk/circuit/rotation.h"
#include "tachyon/zk/plonk/permutation/grand_product_argument.h"

namespace tachyon::zk {

template <typename Poly, typename Evals>
template <typename PCS, typename F>
LookupPermuted<Poly, Evals> LookupArgumentRunner<Poly, Evals>::PermuteArgument(
    ProverBase<PCS>* prover, const LookupArgument<F>& argument, const F& theta,
    const SimpleEvaluator<Evals>& evaluator_tpl) {
  // A_compressed(X) = θᵐ⁻¹A₀(X) + θᵐ⁻²A₁(X) + ... + θAₘ₋₂(X) + Aₘ₋₁(X)
  Evals compressed_input_expression = CompressExpressions(
      prover->domain(), argument.input_expressions(), theta, evaluator_tpl);

  // S_compressed(X) = θᵐ⁻¹S₀(X) + θᵐ⁻²S₁(X) + ... + θSₘ₋₂(X) + Sₘ₋₁(X)
  Evals compressed_table_expression =
      CompressExpressions(prover->domain(), argument.table_expressions(), theta,
                          evaluator_tpl, &compressed_table_expression);

  // Permute compressed (InputExpression, TableExpression) pair.
  LookupPair<Evals> compressed_evals_pair(
      std::move(compressed_input_expression),
      std::move(compressed_table_expression));

  // A'(X), S'(X)
  LookupPair<Evals> permuted_evals_pair;
  CHECK(PermuteExpressionPair(prover, compressed_evals_pair,
                              &permuted_evals_pair));

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

template <typename Poly, typename Evals>
template <typename PCS, typename F>
LookupCommitted<Poly> LookupArgumentRunner<Poly, Evals>::CommitPermuted(
    ProverBase<PCS>* prover, LookupPermuted<Poly, Evals>&& permuted,
    const F& beta, const F& gamma) {
  BlindedPolynomial<Poly> grand_product_poly = GrandProductArgument::Commit(
      prover, CreateNumeratorCallback<F>(permuted, beta, gamma),
      CreateDenominatorCallback<F>(permuted, beta, gamma));

  return LookupCommitted<Poly>(std::move(permuted).TakePermutedInputPoly(),
                               std::move(permuted).TakePermutedTablePoly(),
                               std::move(grand_product_poly));
}

template <typename Poly, typename Evals>
template <typename PCS, typename F>
LookupEvaluated<Poly> LookupArgumentRunner<Poly, Evals>::EvaluateCommitted(
    ProverBase<PCS>* prover, LookupCommitted<Poly>&& committed, const F& x) {
  F x_prev = Rotation::Prev().RotateOmega(prover->domain(), x);
  F x_next = Rotation::Next().RotateOmega(prover->domain(), x);

  BlindedPolynomial<Poly> product_poly = std::move(committed).TakeProductPoly();
  BlindedPolynomial<Poly> permuted_input_poly =
      std::move(committed).TakePermutedInputPoly();
  BlindedPolynomial<Poly> permuted_table_poly =
      std::move(committed).TakePermutedTablePoly();

  CHECK(prover->Evaluate(product_poly.poly(), x));
  CHECK(prover->Evaluate(product_poly.poly(), x_next));
  CHECK(prover->Evaluate(permuted_input_poly.poly(), x));
  CHECK(prover->Evaluate(permuted_input_poly.poly(), x_prev));
  CHECK(prover->Evaluate(permuted_table_poly.poly(), x));

  return {
      std::move(permuted_input_poly),
      std::move(permuted_table_poly),
      std::move(product_poly),
  };
}

template <typename Poly, typename Evals>
template <typename PCS, typename F>
std::vector<ProverQuery<PCS>> LookupArgumentRunner<Poly, Evals>::OpenEvaluated(
    const ProverBase<PCS>* prover, const LookupEvaluated<Poly>& evaluated,
    const F& x) {
  F x_prev = Rotation::Prev().RotateOmega(prover->domain(), x);
  F x_next = Rotation::Next().RotateOmega(prover->domain(), x);

  return {
      ProverQuery<PCS>(x, evaluated.product_poly().ToRef()),
      ProverQuery<PCS>(x, evaluated.permuted_input_poly().ToRef()),
      ProverQuery<PCS>(std::move(x), evaluated.permuted_table_poly().ToRef()),
      ProverQuery<PCS>(std::move(x_prev),
                       evaluated.permuted_input_poly().ToRef()),
      ProverQuery<PCS>(std::move(x_next), evaluated.product_poly().ToRef())};
}

template <typename Poly, typename Evals>
template <typename F>
base::ParallelizeCallback3<F>
LookupArgumentRunner<Poly, Evals>::CreateNumeratorCallback(
    const LookupPermuted<Poly, Evals>& permuted, const F& beta,
    const F& gamma) {
  // (A_compressed(xᵢ) + β) * (S_compressed(xᵢ) + γ)
  return [&beta, &gamma, &permuted](absl::Span<F> chunk, size_t chunk_index,
                                    size_t chunk_size) {
    size_t i = chunk_index * chunk_size;
    for (F& value : chunk) {
      value *= (*permuted.compressed_evals_pair().input()[i] + beta);
      value *= (*permuted.compressed_evals_pair().table()[i] + gamma);
      ++i;
    }
  };
}

template <typename Poly, typename Evals>
template <typename F>
base::ParallelizeCallback3<F>
LookupArgumentRunner<Poly, Evals>::CreateDenominatorCallback(
    const LookupPermuted<Poly, Evals>& permuted, const F& beta,
    const F& gamma) {
  // (A'(xᵢ) + β) * (S'(xᵢ) + γ)
  return [&beta, &gamma, &permuted](absl::Span<F> chunk, size_t chunk_index,
                                    size_t chunk_size) {
    size_t i = chunk_index * chunk_size;
    for (F& value : chunk) {
      value = (*permuted.permuted_evals_pair().input()[i] + beta) *
              (*permuted.permuted_evals_pair().table()[i] + gamma);
      ++i;
    }
  };
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_LOOKUP_LOOKUP_ARGUMENT_RUNNER_IMPL_H_
