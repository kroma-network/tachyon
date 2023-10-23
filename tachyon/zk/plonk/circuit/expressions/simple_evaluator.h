// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_SIMPLE_EVALUATOR_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_SIMPLE_EVALUATOR_H_

#include <vector>

#include "tachyon/zk/plonk/circuit/expressions/advice_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/challenge_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/constant_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/evaluator.h"
#include "tachyon/zk/plonk/circuit/expressions/fixed_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/instance_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/negated_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/product_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/scaled_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/selector_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/sum_expression.h"

namespace tachyon::zk {

template <typename Poly>
class SimpleEvaluator
    : public Evaluator<typename Poly::Field, typename Poly::Field> {
 public:
  using Field = typename Poly::Field;

  SimpleEvaluator(int32_t idx, int32_t size, int32_t rot_scale,
                  const std::vector<Poly>* fixeds,
                  const std::vector<Poly>* advices,
                  const std::vector<Poly>* instances,
                  const std::vector<Field>* challenges)
      : idx_(idx),
        size_(size),
        rot_scale_(rot_scale),
        fixeds_(fixeds),
        advices_(advices),
        instances_(instances),
        challenges_(challenges) {}

  int32_t idx() const { return idx_; }
  int32_t size() const { return size_; }
  int32_t rot_scale() const { return rot_scale_; }

  // Evaluator methods
  Field Evaluate(const Expression<Field>* input) const override {
    switch (input->type()) {
      case ExpressionType::kConstant:
        return input->ToConstant()->value();

      case ExpressionType::kSelector:
        break;

      case ExpressionType::kFixed: {
        const FixedExpression<Field>* fixed_expr = input->ToFixed();
        const FixedQuery& query = fixed_expr->query();
        const Poly& poly = (*fixeds_)[query.column().index()];
        const Field* ret =
            poly[query.rotation().GetIndex(idx_, rot_scale_, size_)];
        if (ret == nullptr) {
          return Field::Zero();
        }
        return *ret;
      }

      case ExpressionType::kAdvice: {
        const AdviceExpression<Field>* advice_expr = input->ToAdvice();
        const AdviceQuery& query = advice_expr->query();
        const Poly& poly = (*advices_)[query.column().index()];
        const Field* ret =
            poly[query.rotation().GetIndex(idx_, rot_scale_, size_)];
        if (ret == nullptr) {
          return Field::Zero();
        }
        return *ret;
      }

      case ExpressionType::kInstance: {
        const InstanceExpression<Field>* instance_expr = input->ToInstance();
        const InstanceQuery& query = instance_expr->query();
        const Poly& poly = (*instances_)[query.column().index()];
        const Field* ret =
            poly[query.rotation().GetIndex(idx_, rot_scale_, size_)];
        if (ret == nullptr) {
          return Field::Zero();
        }
        return *ret;
      }
      case ExpressionType::kChallenge:
        return (*challenges_)[input->ToChallenge()->index()];

      case ExpressionType::kNegated: {
        return -Evaluate(input->ToNegated()->expr());
      }

      case ExpressionType::kSum: {
        const SumExpression<Field>* sum = input->ToSum();
        return Evaluate(sum->left()) + Evaluate(sum->right());
      }

      case ExpressionType::kProduct: {
        const ProductExpression<Field>* product = input->ToProduct();
        return Evaluate(product->left()) * Evaluate(product->right());
      }

      case ExpressionType::kScaled: {
        const ScaledExpression<Field>* scaled = input->ToScaled();
        return Evaluate(scaled->left()) * scaled->right();
      }
    }
    NOTREACHED();
    return Field::Zero();
  }

 private:
  int32_t idx_ = 0;
  int32_t size_ = 0;
  int32_t rot_scale_ = 0;
  // not owned
  const std::vector<Poly>* fixeds_ = nullptr;
  // not owned
  const std::vector<Poly>* advices_ = nullptr;
  // not owned
  const std::vector<Poly>* instances_ = nullptr;
  // not owned
  const std::vector<Field>* challenges_ = nullptr;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_SIMPLE_EVALUATOR_H_
