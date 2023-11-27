// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_SIMPLE_EVALUATOR_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_SIMPLE_EVALUATOR_H_

#include <vector>

#include "absl/types/span.h"

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
#include "tachyon/zk/plonk/circuit/table.h"

namespace tachyon::zk {

template <typename Evals>
class SimpleEvaluator
    : public Evaluator<typename Evals::Field, typename Evals::Field> {
 public:
  using Field = typename Evals::Field;

  SimpleEvaluator() = default;
  SimpleEvaluator(int32_t idx, int32_t size, int32_t rot_scale,
                  const Table<Evals>& table, absl::Span<const Field> challenges)
      : idx_(idx),
        size_(size),
        rot_scale_(rot_scale),
        fixed_columns_(table.fixed_columns()),
        advice_columns_(table.advice_columns()),
        instance_columns_(table.instance_columns()),
        challenges_(challenges) {}

  int32_t idx() const { return idx_; }
  void set_idx(int32_t idx) { idx_ = idx; }
  int32_t size() const { return size_; }
  int32_t rot_scale() const { return rot_scale_; }

  // Evaluator methods
  Field Evaluate(const Expression<Field>* input) override {
    class ScopedIdxIncrement {
     public:
      explicit ScopedIdxIncrement(SimpleEvaluator* evaluator)
          : evaluator(evaluator) {}
      ~ScopedIdxIncrement() { ++evaluator->idx_; }

     private:
      SimpleEvaluator* const evaluator;
    };
    ScopedIdxIncrement scoped_idx_increment(this);

    switch (input->type()) {
      case ExpressionType::kConstant:
        return input->ToConstant()->value();

      case ExpressionType::kSelector:
        break;

      case ExpressionType::kFixed: {
        const FixedExpression<Field>* fixed_expr = input->ToFixed();
        const FixedQuery& query = fixed_expr->query();
        const Evals& evals = fixed_columns_[query.column().index()];
        const Field* ret =
            evals[query.rotation().GetIndex(idx_, rot_scale_, size_)];
        if (ret == nullptr) {
          return Field::Zero();
        }
        return *ret;
      }

      case ExpressionType::kAdvice: {
        const AdviceExpression<Field>* advice_expr = input->ToAdvice();
        const AdviceQuery& query = advice_expr->query();
        const Evals& evals = advice_columns_[query.column().index()];
        const Field* ret =
            evals[query.rotation().GetIndex(idx_, rot_scale_, size_)];
        if (ret == nullptr) {
          return Field::Zero();
        }
        return *ret;
      }

      case ExpressionType::kInstance: {
        const InstanceExpression<Field>* instance_expr = input->ToInstance();
        const InstanceQuery& query = instance_expr->query();
        const Evals& evals = instance_columns_[query.column().index()];
        const Field* ret =
            evals[query.rotation().GetIndex(idx_, rot_scale_, size_)];
        if (ret == nullptr) {
          return Field::Zero();
        }
        return *ret;
      }
      case ExpressionType::kChallenge:
        return challenges_[input->ToChallenge()->challenge().index()];

      case ExpressionType::kNegated:
        return -Evaluate(input->ToNegated()->expr());

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
        return Evaluate(scaled->expr()) * scaled->scale();
      }
    }
    NOTREACHED();
    return Field::Zero();
  }

 private:
  int32_t idx_ = 0;
  int32_t size_ = 0;
  int32_t rot_scale_ = 0;
  absl::Span<const Evals> fixed_columns_;
  absl::Span<const Evals> advice_columns_;
  absl::Span<const Evals> instance_columns_;
  absl::Span<const Field> challenges_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_SIMPLE_EVALUATOR_H_
