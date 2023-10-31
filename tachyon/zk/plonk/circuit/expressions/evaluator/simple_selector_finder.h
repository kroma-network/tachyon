// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_SIMPLE_SELECTOR_FINDER_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_SIMPLE_SELECTOR_FINDER_H_

#include "tachyon/zk/plonk/circuit/expressions/evaluator.h"
#include "tachyon/zk/plonk/circuit/expressions/negated_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/product_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/scaled_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/selector_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/sum_expression.h"

namespace tachyon::zk {

template <typename F>
class SimpleSelectorFinder : public Evaluator<F, bool> {
 public:
  // Evaluator methods
  bool Evaluate(const Expression<F>* input) const override {
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
        return Evaluate(input->ToNegated()->expr());
      case ExpressionType::kSum: {
        const SumExpression<F>* sum = input->ToSum();
        return Evaluate(sum->left()) || Evaluate(sum->right());
      }
      case ExpressionType::kProduct: {
        const ProductExpression<F>* product = input->ToProduct();
        return Evaluate(product->left()) || Evaluate(product->right());
      }
      case ExpressionType::kScaled: {
        const ScaledExpression<F>* scaled = input->ToScaled();
        return Evaluate(scaled->expr());
      }
    }
    NOTREACHED();
    return false;
  }
};

template <typename F>
bool Expression<F>::ContainsSimpleSelector() const {
  SimpleSelectorFinder<F> finder;
  return Evaluate(&finder);
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_SIMPLE_SELECTOR_FINDER_H_
