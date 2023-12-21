// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_LOOKUP_VERIFICATION_H_
#define TACHYON_ZK_LOOKUP_LOOKUP_VERIFICATION_H_

#include <memory>
#include <vector>

#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/lookup/lookup_argument.h"
#include "tachyon/zk/lookup/lookup_verification_data.h"
#include "tachyon/zk/plonk/vanishing/vanishing_verification_evaluator.h"

namespace tachyon::zk {

template <typename F>
F CompressExpressions(
    const std::vector<std::unique_ptr<Expression<F>>>& expressions,
    const F& theta, VanishingVerificationEvaluator<F>& evaluator) {
  F compressed_value = F::Zero();
  for (size_t expr_idx = 0; expr_idx < expressions.size(); ++expr_idx) {
    compressed_value *= theta;
    compressed_value += evaluator.Evaluate(expressions[expr_idx].get());
  }
  return compressed_value;
}

template <typename F, typename C>
F CreateProductExpression(const LookupVerificationData<F, C>& data,
                          const LookupArgument<F>& argument) {
  VanishingVerificationEvaluator<F> evaluator(data);
  // z(ω * X) * (a'(X) + β) * (s'(X) + γ)
  // - z(X) * (θᵐ⁻¹a₀(X) + ... + aₘ₋₁(X) + β) * (θᵐ⁻¹s₀(X) + ... + sₘ₋₁(X) + γ)
  F left = *data.product_next_eval * (*data.permuted_input_eval + *data.beta) *
           (*data.permuted_table_eval + *data.gamma);
  F compressed_input_expression =
      CompressExpressions(argument.input_expressions(), *data.theta, evaluator);
  F compressed_table_expression =
      CompressExpressions(argument.table_expressions(), *data.theta, evaluator);
  F right = *data.product_eval * (compressed_input_expression + *data.beta) *
            (compressed_table_expression + *data.gamma);
  return left - right;
}

constexpr size_t GetSizeOfLookupVerificationExpressions() { return 5; }

template <typename F, typename C>
std::vector<F> CreateLookupVerificationExpressions(
    const LookupVerificationData<F, C>& data,
    const LookupArgument<F>& argument) {
  F active_rows = F::One() - (*data.l_last + *data.l_blind);
  std::vector<F> ret;
  ret.reserve(GetSizeOfLookupVerificationExpressions());
  // l_first(X) * (1 - z'(X)) = 0
  ret.push_back(*data.l_first * (F::One() - *data.product_eval));
  // l_last(X) * (z(X)² - z(X)) = 0
  ret.push_back(*data.l_last *
                (data.product_eval->Square() - *data.product_eval));
  // (1 - (l_last(X) + l_blind(X))) * (
  //  z(ω * X) * (a'(X) + β) * (s'(X) + γ) -
  //  z(X) * (θᵐ⁻¹a₀(X) + ... + aₘ₋₁(X) + β) * (θᵐ⁻¹s₀(X) + ... + sₘ₋₁(X) + γ)
  // ) = 0
  ret.push_back(active_rows * CreateProductExpression(data, argument));
  // l_first(X) * (a'(X) - s'(X)) = 0
  ret.push_back(*data.l_first *
                (*data.permuted_input_eval - *data.permuted_table_eval));
  // (1 - (l_last(X) + l_blind(X))) *
  // (a′(X) − s′(X)) * (a′(X) − a′(ω⁻¹ * X)) = 0
  ret.push_back(active_rows *
                (*data.permuted_input_eval - *data.permuted_table_eval) *
                (*data.permuted_input_eval - *data.permuted_input_inv_eval));
  return ret;
}

constexpr size_t GetSizeOfLookupVerifierQueries() { return 5; }

template <typename PCS, typename F, typename C,
          typename Poly = typename PCS::Poly>
std::vector<crypto::PolynomialOpening<Poly, C>> CreateLookupQueries(
    const LookupVerificationData<F, C>& data) {
  std::vector<crypto::PolynomialOpening<Poly, C>> queries;
  queries.reserve(GetSizeOfLookupVerifierQueries());
  // Open lookup product commitment at x.
  queries.emplace_back(base::DeepRef<const C>(data.product_commitment),
                       base::DeepRef<const F>(data.x), *data.product_eval);
  // Open lookup input commitments at x.
  queries.emplace_back(
      base::DeepRef<const C>(&data.permuted_commitment->input()),
      base::DeepRef<const F>(data.x), *data.permuted_input_eval);
  // Open lookup table commitments at x.
  queries.emplace_back(
      base::DeepRef<const C>(&data.permuted_commitment->table()),
      base::DeepRef<const F>(data.x), *data.permuted_table_eval);
  // Open lookup input commitments at ω⁻¹ * x.
  queries.emplace_back(
      base::DeepRef<const C>(&data.permuted_commitment->input()),
      base::DeepRef<const F>(data.x_prev), *data.permuted_input_inv_eval);
  // Open lookup product commitments at ω * x.
  queries.emplace_back(base::DeepRef<const C>(data.product_commitment),
                       base::DeepRef<const F>(data.x_next),
                       *data.product_next_eval);
  return queries;
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_LOOKUP_LOOKUP_VERIFICATION_H_
