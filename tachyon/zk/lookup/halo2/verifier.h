// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_HALO2_VERIFIER_H_
#define TACHYON_ZK_LOOKUP_HALO2_VERIFIER_H_

#include <memory>
#include <vector>

#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/lookup/halo2/opening_point_set.h"
#include "tachyon/zk/lookup/halo2/verifier_data.h"
#include "tachyon/zk/lookup/lookup_argument.h"
#include "tachyon/zk/lookup/verifying_evaluator.h"
#include "tachyon/zk/plonk/base/l_values.h"

namespace tachyon::zk::lookup::halo2 {

template <typename F, typename C>
class Verifier {
 public:
  explicit Verifier(const VerifierData<F, C>& data) : data_(data) {}

  void Evaluate(const std::vector<lookup::Argument<F>>& arguments,
                const plonk::LValues<F>& l_values, std::vector<F>& evals) {
    lookup::VerifyingEvaluator<F> evaluator(data_);

    F active_rows = F::One() - (l_values.last + l_values.blind);
    for (size_t i = 0; i < data_.grand_product_commitments.size(); ++i) {
      // l_first(x) * (1 - Zₗ,ᵢ(x)) = 0
      evals.push_back(l_values.first *
                      (F::One() - data_.grand_product_evals[i]));
      // l_last(x) * (Zₗ,ᵢ(x)² - Zₗ,ᵢ(x)) = 0
      evals.push_back(l_values.last * (data_.grand_product_evals[i].Square() -
                                       data_.grand_product_evals[i]));
      // (1 - (l_last(x) + l_blind(x))) * (
      //  Zₗ,ᵢ(ω * x) * (A'ᵢ(x) + β) * (S'ᵢ(x) + γ) -
      //  Zₗ,ᵢ(x) * (A_compressedᵢ(x) + β) * (S_compressedᵢ(x) + γ)
      // ) = 0
      evals.push_back(active_rows *
                      CreateGrandProductEvaluation(i, arguments[i], evaluator));
      // l_first(x) * (A'ᵢ(x) - S'ᵢ(x)) = 0
      evals.push_back(l_values.first * (data_.permuted_input_evals[i] -
                                        data_.permuted_table_evals[i]));
      // (1 - (l_last(x) + l_blind(x))) *
      // (A'ᵢ(x) − S'ᵢ(x)) * (A'ᵢ(x) − A'ᵢ(ω⁻¹ * x)) = 0
      evals.push_back(
          active_rows *
          (data_.permuted_input_evals[i] - data_.permuted_table_evals[i]) *
          (data_.permuted_input_evals[i] - data_.permuted_input_prev_evals[i]));
    }
  }

  template <typename Poly>
  void Open(const OpeningPointSet<F>& point_set,
            std::vector<crypto::PolynomialOpening<Poly, C>>& openings) const {
    if (data_.grand_product_commitments.empty()) return;

    base::DeepRef<const F> x_ref(&point_set.x);
    base::DeepRef<const F> x_prev_ref(&point_set.x_prev);
    base::DeepRef<const F> x_next_ref(&point_set.x_next);

#define OPENING(commitment, point, eval) \
  base::Ref<const C>(&data_.commitment), point##_ref, data_.eval

    for (size_t i = 0; i < data_.grand_product_commitments.size(); ++i) {
      openings.emplace_back(
          OPENING(grand_product_commitments[i], x, grand_product_evals[i]));
      openings.emplace_back(
          OPENING(permuted_commitments[i].input(), x, permuted_input_evals[i]));
      openings.emplace_back(
          OPENING(permuted_commitments[i].table(), x, permuted_table_evals[i]));
      openings.emplace_back(OPENING(permuted_commitments[i].input(), x_prev,
                                    permuted_input_prev_evals[i]));
      openings.emplace_back(OPENING(grand_product_commitments[i], x_next,
                                    grand_product_next_evals[i]));
    }

#undef OPENING
  }

 private:
  F CompressExpressions(
      const std::vector<std::unique_ptr<Expression<F>>>& expressions,
      lookup::VerifyingEvaluator<F>& evaluator) const {
    F compressed_value = F::Zero();
    for (const std::unique_ptr<Expression<F>>& expression : expressions) {
      compressed_value *= data_.theta;
      compressed_value += evaluator.Evaluate(expression.get());
    }
    return compressed_value;
  }

  F CreateGrandProductEvaluation(size_t i, const lookup::Argument<F>& argument,
                                 lookup::VerifyingEvaluator<F>& evaluator) {
    // Zₗ,ᵢ(ω * x) * (A'ᵢ(x) + β) * (S'ᵢ(x) + γ)
    // - Zₗ,ᵢ(x) * (A_compressedᵢ(x) + β) * (S_compressedᵢ(x) + γ)
    F left = data_.grand_product_next_evals[i] *
             (data_.permuted_input_evals[i] + data_.beta) *
             (data_.permuted_table_evals[i] + data_.gamma);
    F compressed_input_expression =
        CompressExpressions(argument.input_expressions(), evaluator);
    F compressed_table_expression =
        CompressExpressions(argument.table_expressions(), evaluator);
    F right = data_.grand_product_evals[i] *
              (compressed_input_expression + data_.beta) *
              (compressed_table_expression + data_.gamma);
    return left - right;
  }

  const VerifierData<F, C>& data_;
};

}  // namespace tachyon::zk::lookup::halo2

#endif  // TACHYON_ZK_LOOKUP_HALO2_VERIFIER_H_
