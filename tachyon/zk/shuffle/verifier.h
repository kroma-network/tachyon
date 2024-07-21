// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_SHUFFLE_VERIFIER_H_
#define TACHYON_ZK_SHUFFLE_VERIFIER_H_

#include <memory>
#include <vector>

#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/zk/plonk/base/l_values.h"
#include "tachyon/zk/plonk/expressions/verifying_evaluator.h"
#include "tachyon/zk/shuffle/argument.h"
#include "tachyon/zk/shuffle/opening_point_set.h"
#include "tachyon/zk/shuffle/verifier_data.h"

namespace tachyon::zk::shuffle {

template <typename F, typename C>
class Verifier {
 public:
  explicit Verifier(const VerifierData<F, C>& data) : data_(data) {}

  Verifier(const VerifierData<F, C>& data, const plonk::LValues<F>& l_values)
      : data_(data), l_values_(&l_values) {}

  void Evaluate(const std::vector<Argument<F>>& arguments,
                std::vector<F>& evals) {
    plonk::VerifyingEvaluator<F> evaluator(data_);

    F active_rows = F::One() - (l_values_->last + l_values_->blind);
    for (size_t i = 0; i < data_.grand_product_commitments.size(); ++i) {
      // l_first(x) * (1 - Zₛ,ᵢ(x)) = 0
      evals.push_back(l_values_->first *
                      (F::One() - data_.grand_product_evals[i]));
      // l_last(x) * (Zₛ,ᵢ(x)² - Zₛ,ᵢ(x)) = 0
      evals.push_back(l_values_->last * (data_.grand_product_evals[i].Square() -
                                         data_.grand_product_evals[i]));
      // (1 - (l_last(x) + l_blind(x))) * (
      //  Zₛ,ᵢ(ω * x) * ( (S_compressedᵢ(x) + γ) -
      //  Zₛ,ᵢ(x) * (A_compressedᵢ(x) + γ)
      // ) = 0
      evals.push_back(active_rows *
                      CreateGrandProductEvaluation(i, arguments[i], evaluator));
    }
  }

  template <typename Poly>
  void Open(const OpeningPointSet<F>& point_set,
            std::vector<crypto::PolynomialOpening<Poly, C>>& openings) const {
    if (data_.grand_product_commitments.empty()) return;

#define OPENING(commitment, point, eval) \
  base::Ref<const C>(&data_.commitment), point_set.point, data_.eval

    for (size_t i = 0; i < data_.grand_product_commitments.size(); ++i) {
      openings.emplace_back(
          OPENING(grand_product_commitments[i], x, grand_product_evals[i]));
      openings.emplace_back(OPENING(grand_product_commitments[i], x_next,
                                    grand_product_next_evals[i]));
    }

#undef OPENING
  }

 private:
  F CompressExpressions(
      const std::vector<std::unique_ptr<Expression<F>>>& expressions,
      plonk::VerifyingEvaluator<F>& evaluator) const {
    F compressed_value = F::Zero();
    for (const std::unique_ptr<Expression<F>>& expression : expressions) {
      compressed_value *= data_.theta;
      compressed_value += evaluator.Evaluate(expression.get());
    }
    return compressed_value;
  }

  F CreateGrandProductEvaluation(size_t i, const Argument<F>& argument,
                                 plonk::VerifyingEvaluator<F>& evaluator) {
    // Zₛ,ᵢ(ω * x) * (S_compressedᵢ(x) + γ) - Zₛ,ᵢ(x) * (A_compressedᵢ(x) + γ)
    F compressed_input_expression =
        CompressExpressions(argument.input_expressions(), evaluator);
    F compressed_shuffle_expression =
        CompressExpressions(argument.shuffle_expressions(), evaluator);
    F left = data_.grand_product_next_evals[i] *
             (compressed_shuffle_expression + data_.gamma);
    F right = data_.grand_product_evals[i] *
              (compressed_input_expression + data_.gamma);
    return left - right;
  }

  VerifierData<F, C> data_;
  const plonk::LValues<F>* l_values_ = nullptr;
};

}  // namespace tachyon::zk::shuffle

#endif  // TACHYON_ZK_SHUFFLE_VERIFIER_H_
