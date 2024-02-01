// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_EXPRESSIONS_EVALUATOR_SELECTOR_REPLACER_H_
#define TACHYON_ZK_EXPRESSIONS_EVALUATOR_SELECTOR_REPLACER_H_

#include <memory>
#include <vector>

#include "tachyon/zk/expressions/evaluator.h"
#include "tachyon/zk/expressions/negated_expression.h"
#include "tachyon/zk/expressions/product_expression.h"
#include "tachyon/zk/expressions/scaled_expression.h"
#include "tachyon/zk/expressions/selector_expression.h"
#include "tachyon/zk/expressions/sum_expression.h"

namespace tachyon::zk {

template <typename F>
class SelectorsReplacer : public Evaluator<F, std::unique_ptr<Expression<F>>> {
 public:
  SelectorsReplacer(
      const std::vector<base::Ref<const Expression<F>>>& replacements,
      bool must_be_non_simple)
      : replacements_(replacements), must_be_non_simple_(must_be_non_simple) {}

  // Evaluator methods
  std::unique_ptr<Expression<F>> Evaluate(const Expression<F>* input) override {
    switch (input->type()) {
      case ExpressionType::kConstant:
      case ExpressionType::kFixed:
      case ExpressionType::kAdvice:
      case ExpressionType::kInstance:
      case ExpressionType::kChallenge:
        return input->Clone();
      case ExpressionType::kSelector: {
        const plonk::Selector& selector = input->ToSelector()->selector();
        if (must_be_non_simple_) {
          // Simple selectors are prohibited from appearing in
          // expressions in the lookup argument by |ConstraintSystem|.
          CHECK(!selector.is_simple());
        }
        return replacements_[selector.index()]->Clone();
      }
      case ExpressionType::kNegated:
        return ExpressionFactory<F>::Negated(
            Evaluate(input->ToNegated()->expr()));
      case ExpressionType::kSum: {
        const SumExpression<F>* sum = input->ToSum();
        return ExpressionFactory<F>::Sum(Evaluate(sum->left()),
                                         Evaluate(sum->right()));
      }
      case ExpressionType::kProduct: {
        const ProductExpression<F>* product = input->ToProduct();
        return ExpressionFactory<F>::Product(Evaluate(product->left()),
                                             Evaluate(product->right()));
      }
      case ExpressionType::kScaled: {
        const ScaledExpression<F>* scaled = input->ToScaled();
        return ExpressionFactory<F>::Scaled(Evaluate(scaled->expr()),
                                            scaled->scale());
      }
    }
    NOTREACHED();
    return nullptr;
  }

 private:
  const std::vector<base::Ref<const Expression<F>>>& replacements_;
  const bool must_be_non_simple_;
};

template <typename F>
std::unique_ptr<Expression<F>> Expression<F>::ReplaceSelectors(
    const std::vector<base::Ref<const Expression<F>>>& replacements,
    bool must_be_non_simple) const {
  SelectorsReplacer<F> replacer(replacements, must_be_non_simple);
  return Evaluate(&replacer);
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_EXPRESSIONS_EVALUATOR_SELECTOR_REPLACER_H_
