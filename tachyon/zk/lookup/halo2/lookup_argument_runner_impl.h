// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_HALO2_LOOKUP_ARGUMENT_RUNNER_IMPL_H_
#define TACHYON_ZK_LOOKUP_HALO2_LOOKUP_ARGUMENT_RUNNER_IMPL_H_

#include <utility>
#include <vector>

#include "tachyon/base/ref.h"
#include "tachyon/zk/base/rotation.h"
#include "tachyon/zk/lookup/halo2/compress_expression.h"
#include "tachyon/zk/lookup/halo2/lookup_argument_runner.h"
#include "tachyon/zk/lookup/halo2/permute_expression_pair.h"
#include "tachyon/zk/plonk/permutation/grand_product_argument.h"

namespace tachyon::zk::lookup::halo2 {

// static
template <typename Poly, typename Evals>
template <typename PCS>
LookupPermuted<Poly, Evals> LookupArgumentRunner<Poly, Evals>::PermuteArgument(
    ProverBase<PCS>* prover, const LookupArgument<F>& argument, const F& theta,
    const SimpleEvaluator<Evals>& evaluator_tpl) {
  // A_compressed(X) = θᵐ⁻¹A₀(X) + θᵐ⁻²A₁(X) + ... + θAₘ₋₂(X) + Aₘ₋₁(X)
  Evals compressed_input_expression = CompressExpressions(
      prover->domain(), argument.input_expressions(), theta, evaluator_tpl);

  // S_compressed(X) = θᵐ⁻¹S₀(X) + θᵐ⁻²S₁(X) + ... + θSₘ₋₂(X) + Sₘ₋₁(X)
  Evals compressed_table_expression = CompressExpressions(
      prover->domain(), argument.table_expressions(), theta, evaluator_tpl);

  // Permute compressed (InputExpression, TableExpression) pair.
  LookupPair<Evals> compressed_evals_pair(
      std::move(compressed_input_expression),
      std::move(compressed_table_expression));

  // A'(X), S'(X)
  LookupPair<Evals> permuted_evals_pair;
  CHECK(PermuteExpressionPair(prover, compressed_evals_pair,
                              &permuted_evals_pair));

  // Commit(A'(X))
  BlindedPolynomial<Poly> permuted_input_poly =
      prover->CommitAndWriteToProofWithBlind(permuted_evals_pair.input());

  // Commit(S'(X))
  BlindedPolynomial<Poly> permuted_table_poly =
      prover->CommitAndWriteToProofWithBlind(permuted_evals_pair.table());

  return {std::move(compressed_evals_pair), std::move(permuted_evals_pair),
          std::move(permuted_input_poly), std::move(permuted_table_poly)};
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
LookupCommitted<Poly> LookupArgumentRunner<Poly, Evals>::CommitPermuted(
    ProverBase<PCS>* prover, LookupPermuted<Poly, Evals>&& permuted,
    const F& beta, const F& gamma) {
  BlindedPolynomial<Poly> grand_product_poly =
      plonk::GrandProductArgument::Commit(
          prover, CreateNumeratorCallback(permuted, beta, gamma),
          CreateDenominatorCallback(permuted, beta, gamma));

  return LookupCommitted<Poly>(std::move(permuted).TakePermutedInputPoly(),
                               std::move(permuted).TakePermutedTablePoly(),
                               std::move(grand_product_poly));
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
LookupEvaluated<Poly> LookupArgumentRunner<Poly, Evals>::EvaluateCommitted(
    ProverBase<PCS>* prover, LookupCommitted<Poly>&& committed, const F& x) {
  F x_prev = Rotation::Prev().RotateOmega(prover->domain(), x);
  F x_next = Rotation::Next().RotateOmega(prover->domain(), x);

  BlindedPolynomial<Poly> product_poly = std::move(committed).TakeProductPoly();
  BlindedPolynomial<Poly> permuted_input_poly =
      std::move(committed).TakePermutedInputPoly();
  BlindedPolynomial<Poly> permuted_table_poly =
      std::move(committed).TakePermutedTablePoly();

  prover->EvaluateAndWriteToProof(product_poly.poly(), x);
  prover->EvaluateAndWriteToProof(product_poly.poly(), x_next);
  prover->EvaluateAndWriteToProof(permuted_input_poly.poly(), x);
  prover->EvaluateAndWriteToProof(permuted_input_poly.poly(), x_prev);
  prover->EvaluateAndWriteToProof(permuted_table_poly.poly(), x);

  return {
      std::move(permuted_input_poly),
      std::move(permuted_table_poly),
      std::move(product_poly),
  };
}

// static
template <typename Poly, typename Evals>
template <typename PCS>
std::vector<crypto::PolynomialOpening<Poly>>
LookupArgumentRunner<Poly, Evals>::OpenEvaluated(
    const ProverBase<PCS>* prover, const LookupEvaluated<Poly>& evaluated,
    const F& x, PointSet<F>& points) {
  F x_prev = Rotation::Prev().RotateOmega(prover->domain(), x);
  F x_next = Rotation::Next().RotateOmega(prover->domain(), x);
  base::DeepRef<const F> x_ref(&x);
  base::DeepRef<const F> x_prev_ref = points.Insert(x_prev);
  base::DeepRef<const F> x_next_ref = points.Insert(x_next);

  return {
      crypto::PolynomialOpening<Poly>(
          base::Ref<const Poly>(&evaluated.product_poly().poly()), x_ref,
          evaluated.product_poly().poly().Evaluate(x)),
      crypto::PolynomialOpening<Poly>(
          base::Ref<const Poly>(&evaluated.permuted_input_poly().poly()), x_ref,
          evaluated.permuted_input_poly().poly().Evaluate(x)),
      crypto::PolynomialOpening<Poly>(
          base::Ref<const Poly>(&evaluated.permuted_table_poly().poly()), x_ref,
          evaluated.permuted_table_poly().poly().Evaluate(x)),
      crypto::PolynomialOpening<Poly>(
          base::Ref<const Poly>(&evaluated.permuted_input_poly().poly()),
          x_prev_ref, evaluated.permuted_input_poly().poly().Evaluate(x_prev)),
      crypto::PolynomialOpening<Poly>(
          base::Ref<const Poly>(&evaluated.product_poly().poly()), x_next_ref,
          evaluated.product_poly().poly().Evaluate(x_next))};
}

// static
template <typename Poly, typename Evals>
base::ParallelizeCallback3<typename Poly::Field>
LookupArgumentRunner<Poly, Evals>::CreateNumeratorCallback(
    const LookupPermuted<Poly, Evals>& permuted, const F& beta,
    const F& gamma) {
  // (A_compressed(xᵢ) + β) * (S_compressed(xᵢ) + γ)
  return [&beta, &gamma, &permuted](absl::Span<F> chunk, size_t chunk_index,
                                    size_t chunk_size) {
    size_t i = chunk_index * chunk_size;
    for (F& value : chunk) {
      value *= (permuted.compressed_evals_pair().input()[i] + beta);
      value *= (permuted.compressed_evals_pair().table()[i] + gamma);
      ++i;
    }
  };
}

// static
template <typename Poly, typename Evals>
base::ParallelizeCallback3<typename Poly::Field>
LookupArgumentRunner<Poly, Evals>::CreateDenominatorCallback(
    const LookupPermuted<Poly, Evals>& permuted, const F& beta,
    const F& gamma) {
  // (A'(xᵢ) + β) * (S'(xᵢ) + γ)
  return [&beta, &gamma, &permuted](absl::Span<F> chunk, size_t chunk_index,
                                    size_t chunk_size) {
    size_t i = chunk_index * chunk_size;
    for (F& value : chunk) {
      value = (permuted.permuted_evals_pair().input()[i] + beta) *
              (permuted.permuted_evals_pair().table()[i] + gamma);
      ++i;
    }
  };
}

}  // namespace tachyon::zk::lookup::halo2

#endif  // TACHYON_ZK_LOOKUP_HALO2_LOOKUP_ARGUMENT_RUNNER_IMPL_H_
