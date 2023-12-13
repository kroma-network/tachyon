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
template <typename PCSTy, typename F>
LookupPermuted<Poly, Evals> LookupArgumentRunner<Poly, Evals>::PermuteArgument(
    Prover<PCSTy>* prover, const LookupArgument<F>& argument, const F& theta,
    const SimpleEvaluator<Evals>& evaluator_tpl) {
  // A_compressed(X) = θᵐ⁻¹A₀(X) + θᵐ⁻²A₁(X) + ... + θAₘ₋₂(X) + Aₘ₋₁(X)
  Evals compressed_input_expression;
  CHECK(CompressExpressions(prover->domain(), argument.input_expressions(),
                            theta, evaluator_tpl,
                            &compressed_input_expression));

  // S_compressed(X) = θᵐ⁻¹S₀(X) + θᵐ⁻²S₁(X) + ... + θSₘ₋₂(X) + Sₘ₋₁(X)
  Evals compressed_table_expression;
  CHECK(CompressExpressions(prover->domain(), argument.table_expressions(),
                            theta, evaluator_tpl,
                            &compressed_table_expression));

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
template <typename PCSTy, typename F>
LookupCommitted<Poly> LookupArgumentRunner<Poly, Evals>::CommitPermuted(
    Prover<PCSTy>* prover, LookupPermuted<Poly, Evals>&& permuted,
    const F& beta, const F& gamma) {
  BlindedPolynomial<Poly> grand_product_poly = GrandProductArgument::Commit(
      prover, CreateNumeratorCallback<F>(permuted, beta, gamma),
      CreateDenominatorCallback<F>(permuted, beta, gamma));

  return LookupCommitted<Poly>(std::move(permuted).permuted_input_poly(),
                               std::move(permuted).permuted_table_poly(),
                               std::move(grand_product_poly));
}

template <typename Poly, typename Evals>
template <typename PCSTy, typename F>
LookupEvaluated<Poly> LookupArgumentRunner<Poly, Evals>::EvaluateCommitted(
    Prover<PCSTy>* prover, LookupCommitted<Poly>&& committed, const F& x) {
  F x_inv = Rotation::Prev().RotateOmega(prover->domain(), x);
  F x_next = Rotation::Next().RotateOmega(prover->domain(), x);

  BlindedPolynomial<Poly> product_poly = std::move(committed).product_poly();
  BlindedPolynomial<Poly> permuted_input_poly =
      std::move(committed).permuted_input_poly();
  BlindedPolynomial<Poly> permuted_table_poly =
      std::move(committed).permuted_table_poly();

  prover->Evaluate(product_poly.poly(), x);
  prover->Evaluate(product_poly.poly(), x_next);
  prover->Evaluate(permuted_input_poly.poly(), x);
  prover->Evaluate(permuted_input_poly.poly(), x_inv);
  prover->Evaluate(permuted_table_poly.poly(), x);

  return {
      std::move(permuted_input_poly),
      std::move(permuted_table_poly),
      std::move(product_poly),
  };
}

template <typename Poly, typename Evals>
template <typename PCSTy, typename F>
std::vector<ProverQuery<PCSTy>>
LookupArgumentRunner<Poly, Evals>::OpenEvaluated(
    const Prover<PCSTy>* prover, const LookupEvaluated<Poly>& evaluated,
    const F& x) {
  F x_inv = Rotation::Prev().RotateOmega(prover->domain(), x);
  F x_next = Rotation::Next().RotateOmega(prover->domain(), x);

  return {
      ProverQuery<PCSTy>(x, evaluated.product_poly().ToRef()),
      ProverQuery<PCSTy>(x, evaluated.permuted_input_poly().ToRef()),
      ProverQuery<PCSTy>(std::move(x), evaluated.permuted_table_poly().ToRef()),
      ProverQuery<PCSTy>(std::move(x_inv),
                         evaluated.permuted_input_poly().ToRef()),
      ProverQuery<PCSTy>(std::move(x_next), evaluated.product_poly().ToRef())};
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
