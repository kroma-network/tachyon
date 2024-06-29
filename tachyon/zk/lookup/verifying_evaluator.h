// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_LOOKUP_VERIFYING_EVALUATOR_H_
#define TACHYON_ZK_LOOKUP_VERIFYING_EVALUATOR_H_

#include "tachyon/zk/expressions/constant_expression.h"
#include "tachyon/zk/expressions/evaluator.h"
#include "tachyon/zk/expressions/negated_expression.h"
#include "tachyon/zk/expressions/product_expression.h"
#include "tachyon/zk/expressions/scaled_expression.h"
#include "tachyon/zk/expressions/sum_expression.h"
#include "tachyon/zk/plonk/base/multi_phase_evaluations.h"
#include "tachyon/zk/plonk/expressions/advice_expression.h"
#include "tachyon/zk/plonk/expressions/challenge_expression.h"
#include "tachyon/zk/plonk/expressions/fixed_expression.h"
#include "tachyon/zk/plonk/expressions/instance_expression.h"
#include "tachyon/zk/plonk/expressions/selector_expression.h"

namespace tachyon::zk::lookup {

template <typename F>
class VerifyingEvaluator : public Evaluator<F, F> {
 public:
  explicit VerifyingEvaluator(const plonk::MultiPhaseEvaluations<F>& data)
      : data_(data) {}

  // Evaluator methods
  F Evaluate(const Expression<F>* input) override {
    switch (input->type()) {
      case ExpressionType::kConstant:
        return input->ToConstant()->value();
      case ExpressionType::kSelector:
        NOTREACHED() << "virtual selectors are removed during optimization";
        break;
      case ExpressionType::kFixed: {
        const plonk::FixedExpression<F>* fixed_expr = input->ToFixed();
        const plonk::FixedQuery& query = fixed_expr->query();
        return data_.fixed_evals[query.index()];
      }
      case ExpressionType::kAdvice: {
        const plonk::AdviceExpression<F>* advice_expr = input->ToAdvice();
        const plonk::AdviceQuery& query = advice_expr->query();
        return data_.advice_evals[query.index()];
      }
      case ExpressionType::kInstance: {
        const plonk::InstanceExpression<F>* instance_expr = input->ToInstance();
        const plonk::InstanceQuery& query = instance_expr->query();
        return data_.instance_evals[query.index()];
      }
      case ExpressionType::kChallenge: {
        const plonk::ChallengeExpression<F>* challenge_expr =
            input->ToChallenge();
        plonk::Challenge challenge = challenge_expr->challenge();
        return data_.challenges[challenge.index()];
      }
      case ExpressionType::kNegated: {
        const NegatedExpression<F>* negated_expr = input->ToNegated();
        return -Evaluate(negated_expr->expr());
      }
      case ExpressionType::kSum: {
        const SumExpression<F>* sum_expr = input->ToSum();
        return Evaluate(sum_expr->left()) + Evaluate(sum_expr->right());
      }
      case ExpressionType::kProduct: {
        const ProductExpression<F>* product_expr = input->ToProduct();
        return Evaluate(product_expr->left()) * Evaluate(product_expr->right());
      }
      case ExpressionType::kScaled: {
        const ScaledExpression<F>* scaled_expr = input->ToScaled();
        return Evaluate(scaled_expr->expr()) * scaled_expr->scale();
      }
    }
    NOTREACHED();
    return F::Zero();
  }

 private:
  plonk::MultiPhaseEvaluations<F> data_;
};

}  // namespace tachyon::zk::lookup

#endif  // TACHYON_ZK_LOOKUP_VERIFYING_EVALUATOR_H_
