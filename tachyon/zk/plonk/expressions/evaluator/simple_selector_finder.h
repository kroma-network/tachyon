// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_SIMPLE_SELECTOR_FINDER_H_
#define TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_SIMPLE_SELECTOR_FINDER_H_

#include "tachyon/zk/expressions/evaluator.h"
#include "tachyon/zk/expressions/negated_expression.h"
#include "tachyon/zk/expressions/product_expression.h"
#include "tachyon/zk/expressions/scaled_expression.h"
#include "tachyon/zk/expressions/sum_expression.h"
#include "tachyon/zk/plonk/expressions/selector_expression.h"

namespace tachyon::zk::plonk {

template <typename F>
bool ContainsSimpleSelector(const Expression<F>* input) {
  switch (input->type()) {
    case ExpressionType::kConstant:
      return false;
    case ExpressionType::kSelector:
      return input->ToSelector()->selector().is_simple();
    case ExpressionType::kFixed:
      return false;
    case ExpressionType::kAdvice:
      return false;
    case ExpressionType::kInstance:
      return false;
    case ExpressionType::kChallenge:
      return false;
    case ExpressionType::kNegated:
      return ContainsSimpleSelector(input->ToNegated()->expr());
    case ExpressionType::kSum: {
      const SumExpression<F>* sum = input->ToSum();
      return ContainsSimpleSelector(sum->left()) ||
             ContainsSimpleSelector(sum->right());
    }
    case ExpressionType::kProduct: {
      const ProductExpression<F>* product = input->ToProduct();
      return ContainsSimpleSelector(product->left()) ||
             ContainsSimpleSelector(product->right());
    }
    case ExpressionType::kScaled: {
      const ScaledExpression<F>* scaled = input->ToScaled();
      return ContainsSimpleSelector(scaled->expr());
    }
    case ExpressionType::kFirstRow:
    case ExpressionType::kLastRow:
    case ExpressionType::kTransition:
    case ExpressionType::kVariable:
      NOTREACHED() << "AIR expression " << ExpressionTypeToString(input->type())
                   << " is not allowed in plonk!";
  }
  NOTREACHED();
  return false;
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_SIMPLE_SELECTOR_FINDER_H_
