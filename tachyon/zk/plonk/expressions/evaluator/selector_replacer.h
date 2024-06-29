// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_SELECTOR_REPLACER_H_
#define TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_SELECTOR_REPLACER_H_

#include <memory>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/base/ref.h"
#include "tachyon/zk/expressions/evaluator.h"
#include "tachyon/zk/expressions/negated_expression.h"
#include "tachyon/zk/expressions/product_expression.h"
#include "tachyon/zk/expressions/scaled_expression.h"
#include "tachyon/zk/expressions/sum_expression.h"
#include "tachyon/zk/plonk/expressions/selector_expression.h"

namespace tachyon::zk::plonk {

template <typename F>
std::unique_ptr<Expression<F>> ReplaceSelectors(
    const Expression<F>* input,
    const std::vector<base::Ref<const Expression<F>>>& replacements,
    bool must_be_non_simple) {
  switch (input->type()) {
    case ExpressionType::kConstant:
    case ExpressionType::kFixed:
    case ExpressionType::kAdvice:
    case ExpressionType::kInstance:
    case ExpressionType::kChallenge:
      return input->Clone();
    case ExpressionType::kSelector: {
      Selector selector = input->ToSelector()->selector();
      if (must_be_non_simple) {
        // Simple selectors are prohibited from appearing in
        // expressions in the lookup argument by |ConstraintSystem|.
        if (selector.is_simple()) {
          LOG(DFATAL) << "Simple selector found in lookup argument";
        }
      }
      return replacements[selector.index()]->Clone();
    }
    case ExpressionType::kNegated:
      return ExpressionFactory<F>::Negated(ReplaceSelectors(
          input->ToNegated()->expr(), replacements, must_be_non_simple));
    case ExpressionType::kSum: {
      const SumExpression<F>* sum = input->ToSum();
      return ExpressionFactory<F>::Sum(
          ReplaceSelectors(sum->left(), replacements, must_be_non_simple),
          ReplaceSelectors(sum->right(), replacements, must_be_non_simple));
    }
    case ExpressionType::kProduct: {
      const ProductExpression<F>* product = input->ToProduct();
      return ExpressionFactory<F>::Product(
          ReplaceSelectors(product->left(), replacements, must_be_non_simple),
          ReplaceSelectors(product->right(), replacements, must_be_non_simple));
    }
    case ExpressionType::kScaled: {
      const ScaledExpression<F>* scaled = input->ToScaled();
      return ExpressionFactory<F>::Scaled(
          ReplaceSelectors(scaled->expr(), replacements, must_be_non_simple),
          scaled->scale());
    }
    case ExpressionType::kFirstRow:
    case ExpressionType::kLastRow:
    case ExpressionType::kTransition:
    case ExpressionType::kVariable:
      NOTREACHED() << "AIR expression " << ExpressionTypeToString(input->type())
                   << " is not allowed in plonk!";
  }
  NOTREACHED();
  return nullptr;
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_SELECTOR_REPLACER_H_
