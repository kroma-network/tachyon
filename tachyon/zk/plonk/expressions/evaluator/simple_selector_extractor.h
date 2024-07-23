// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_SIMPLE_SELECTOR_EXTRACTOR_H_
#define TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_SIMPLE_SELECTOR_EXTRACTOR_H_

#include <optional>

#include "tachyon/zk/expressions/evaluator.h"
#include "tachyon/zk/expressions/negated_expression.h"
#include "tachyon/zk/expressions/product_expression.h"
#include "tachyon/zk/expressions/scaled_expression.h"
#include "tachyon/zk/expressions/sum_expression.h"
#include "tachyon/zk/plonk/expressions/selector_expression.h"

namespace tachyon::zk::plonk {

template <typename F>
class SimpleSelectorExtractor : public Evaluator<F, std::optional<Selector>> {
 public:
  // Evaluator methods
  std::optional<Selector> Evaluate(const Expression<F>* input) override {
    switch (input->type()) {
      case ExpressionType::kConstant:
        return std::nullopt;
      case ExpressionType::kSelector: {
        Selector selector = input->ToSelector()->selector();
        if (selector.is_simple()) {
          return selector;
        }
        return std::nullopt;
      }
      case ExpressionType::kFixed:
        return std::nullopt;
      case ExpressionType::kAdvice:
        return std::nullopt;
      case ExpressionType::kInstance:
        return std::nullopt;
      case ExpressionType::kChallenge:
        return std::nullopt;
      case ExpressionType::kNegated: {
        return Evaluate(input->ToNegated()->expr());
      }
      case ExpressionType::kSum: {
        const SumExpression<F>* sum = input->ToSum();
        return Pick(Evaluate(sum->left()), Evaluate(sum->right()));
      }
      case ExpressionType::kProduct: {
        const ProductExpression<F>* product = input->ToProduct();
        return Pick(Evaluate(product->left()), Evaluate(product->right()));
      }
      case ExpressionType::kScaled: {
        const ScaledExpression<F>* scaled = input->ToScaled();
        return Evaluate(scaled->expr());
      }
      case ExpressionType::kFirstRow:
      case ExpressionType::kLastRow:
      case ExpressionType::kTransition:
      case ExpressionType::kVariable:
        NOTREACHED() << "AIR expression "
                     << ExpressionTypeToString(input->type())
                     << " is not allowed in plonk!";
    }
    NOTREACHED();
    return std::nullopt;
  }

 private:
  static std::optional<Selector> Pick(const std::optional<Selector>& left,
                                      const std::optional<Selector>& right) {
    CHECK(!(left.has_value() && right.has_value()))
        << "two simple selectors cannot be in the same expression";
    if (left.has_value()) return left;
    if (right.has_value()) return right;
    return std::nullopt;
  }
};

template <typename F>
std::optional<Selector> ExtractSimpleSelector(const Expression<F>* input) {
  SimpleSelectorExtractor<F> extractor;
  return extractor.Evaluate(input);
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_SIMPLE_SELECTOR_EXTRACTOR_H_
