// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_GRAPH_EVALUATOR_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_GRAPH_EVALUATOR_H_

#include <string>
#include <vector>

#include "absl/strings/substitute.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/zk/plonk/circuit/expressions/advice_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/challenge_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/constant_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/evaluator.h"
#include "tachyon/zk/plonk/circuit/expressions/evaluator/calculation.h"
#include "tachyon/zk/plonk/circuit/expressions/fixed_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/instance_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/negated_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/product_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/scaled_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/selector_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/sum_expression.h"

namespace tachyon::zk {

struct TACHYON_EXPORT CalculationInfo {
  Calculation calculation;
  size_t target;

  CalculationInfo() = default;
  CalculationInfo(Calculation calculation, size_t target)
      : calculation(calculation), target(target) {}

  bool operator==(const CalculationInfo& other) const {
    return calculation == other.calculation && target == other.target;
  }
  bool operator!=(const CalculationInfo& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute("{calculation: $0, target: $1}",
                            calculation.ToString(), target);
  }
};

template <typename F>
struct EvaluationData {
  std::vector<F> intermediates;
  std::vector<size_t> rotations;
};

template <typename F>
class GraphEvaluator : public Evaluator<F, ValueSource> {
 public:
  GraphEvaluator() = default;

  const std::vector<F>& constants() const { return constants_; }
  const std::vector<int32_t>& rotations() { return rotations_; }
  const std::vector<CalculationInfo>& calculations() { return calculations_; }
  size_t num_intermediates() const { return num_intermediates_; }

  template <typename Poly>
  F Evaluate(EvaluationData<F>& evaluation_data,
             const ValueSourceData<Poly>& source_data, size_t idx,
             int32_t scale, int32_t size) const {
    for (size_t i = 0; i < rotations_.size(); ++i) {
      evaluation_data.rotations[i] =
          Rotation(rotations_[i]).GetIndex(idx, scale, size);
    }

    for (const CalculationInfo& calculation : calculations_) {
      evaluation_data.intermediates[calculation.target] =
          calculation.calculation.Evaluate(source_data);
    }

    if (calculations_.empty()) return F::Zero();
    return evaluation_data.intermediates[calculations_.back().target];
  }

  // Evaluator methods
  ValueSource Evaluate(const Expression<F>* input) override {
    switch (input->type()) {
      case ExpressionType::kConstant:
        return AddConstant(input->ToConstant()->value());

      case ExpressionType::kSelector:
        break;

      case ExpressionType::kFixed: {
        const FixedExpression<F>* expr = input->ToFixed();
        const FixedQuery& query = expr->query();
        size_t rotation_index = AddRotation(query.rotation());
        return AddCalculation(Calculation::Store(
            ValueSource::Fixed(query.column().index(), rotation_index)));
      }

      case ExpressionType::kAdvice: {
        const AdviceExpression<F>* expr = input->ToAdvice();
        const AdviceQuery& query = expr->query();
        size_t rotation_index = AddRotation(query.rotation());
        return AddCalculation(Calculation::Store(
            ValueSource::Advice(query.column().index(), rotation_index)));
      }

      case ExpressionType::kInstance: {
        const InstanceExpression<F>* expr = input->ToInstance();
        const InstanceQuery& query = expr->query();
        size_t rotation_index = AddRotation(query.rotation());
        return AddCalculation(Calculation::Store(
            ValueSource::Instance(query.column().index(), rotation_index)));
      }

      case ExpressionType::kChallenge: {
        const ChallengeExpression<F>* expr = input->ToChallenge();
        const Challenge& challenge = expr->challenge();
        return AddCalculation(
            Calculation::Store(ValueSource::Challenge(challenge.index())));
      }

      case ExpressionType::kNegated: {
        const NegatedExpression<F>* expr = input->ToNegated();
        if (expr->expr()->type() == ExpressionType::kConstant) {
          return AddConstant(-expr->expr()->ToConstant()->value());
        }
        ValueSource result = AddExpression(expr->expr());
        if (result.IsZeroConstant()) {
          return result;
        }
        return AddCalculation(Calculation::Negate(result));
      }

      case ExpressionType::kSum: {
        const SumExpression<F>* sum = input->ToSum();
        ValueSource left_result = AddExpression(sum->left());
        if (sum->right()->type() == ExpressionType::kNegated) {
          const NegatedExpression<F>* right = sum->right()->ToNegated();
          ValueSource right_result = AddExpression(right->expr());
          if (left_result.IsZeroConstant()) {
            return AddCalculation(Calculation::Negate(right_result));
          } else if (right_result.IsZeroConstant()) {
            return left_result;
          } else {
            return AddCalculation(Calculation::Sub(left_result, right_result));
          }
        } else {
          ValueSource right_result = AddExpression(sum->right());
          if (left_result.IsZeroConstant()) {
            return right_result;
          } else if (right_result.IsZeroConstant()) {
            return left_result;
          } else {
            // NOTE(chokobole): I don't know why this is needed.
            // See
            // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/evaluation.rs#L709-L713.
            if (left_result <= right_result) {
              return AddCalculation(
                  Calculation::Add(left_result, right_result));
            } else {
              return AddCalculation(
                  Calculation::Add(right_result, left_result));
            }
          }
        }
      }

      case ExpressionType::kProduct: {
        const ProductExpression<F>* product = input->ToProduct();
        ValueSource left_result = AddExpression(product->left());
        ValueSource right_result = AddExpression(product->right());
        if (left_result.IsZeroConstant() || right_result.IsZeroConstant()) {
          return ValueSource::ZeroConstant();
        } else if (left_result.IsOneConstant()) {
          return right_result;
        } else if (right_result.IsOneConstant()) {
          return left_result;
        } else if (left_result.IsTwoConstant()) {
          return AddCalculation(Calculation::Double(right_result));
        } else if (right_result.IsTwoConstant()) {
          return AddCalculation(Calculation::Double(left_result));
        } else if (left_result == right_result) {
          return AddCalculation(Calculation::Square(left_result));
        } else {
          // NOTE(chokobole): we can do |left_result < right_result|.
          // But ValueSource only implements <= operator and original source
          // code also compares like below, too. See
          // https://github.com/kroma-network/halo2/blob/7d0a36990452c8e7ebd600de258420781a9b7917/halo2_proofs/src/plonk/evaluation.rs#L732
          if (left_result <= right_result) {
            return AddCalculation(Calculation::Mul(left_result, right_result));
          } else {
            return AddCalculation(Calculation::Mul(right_result, left_result));
          }
        }
      }

      case ExpressionType::kScaled: {
        const ScaledExpression<F>* scaled = input->ToScaled();
        const F& scale = scaled->scale();
        if (scale.IsZero()) {
          return ValueSource::ZeroConstant();
        } else if (scale.IsOne()) {
          return AddExpression(scaled->expr());
        } else {
          ValueSource constant = AddConstant(scale);
          ValueSource result = AddExpression(scaled->expr());
          return AddCalculation(Calculation::Mul(result, constant));
        }
      }
    }
    NOTREACHED();
    return ValueSource();
  }

  EvaluationData<F> CreateInstance() const {
    return {
        base::CreateVector(num_intermediates_, F::Zero()),
        base::CreateVector(rotations_.size(), size_t{0}),
    };
  }

 private:
  size_t AddRotation(const Rotation& rotation) {
    size_t rotation_index = rotation.value();
    std::optional<size_t> position = base::FindIndexIf(
        rotations_,
        [rotation_index](size_t value) { return rotation_index == value; });
    if (position.has_value()) return position.value();
    rotations_.push_back(rotation_index);
    return rotations_.size() - 1;
  }

  ValueSource AddConstant(const F& constant) {
    std::optional<size_t> position = base::FindIndex(constants_, constant);
    if (position.has_value()) return ValueSource::Constant(position.value());
    constants_.push_back(constant);
    return ValueSource::Constant(constants_.size() - 1);
  }

  // Currently does the simplest thing possible: just stores the
  // resulting value so the result can be reused  when that calculation
  // is done multiple times.
  ValueSource AddCalculation(const Calculation& calculation) {
    auto it = std::find_if(calculations_.begin(), calculations_.end(),
                           [&calculation](const CalculationInfo& info) {
                             return info.calculation == calculation;
                           });
    if (it != calculations_.end()) return ValueSource::Intermediate(it->target);
    size_t target = num_intermediates_++;
    calculations_.emplace_back(calculation, target);
    return ValueSource::Intermediate(target);
  }

  // Generates an optimized evaluation for the expression
  ValueSource AddExpression(const Expression<F>* expression) {
    return expression->Evaluate(this);
  }

  std::vector<F> constants_ = {F::Zero(), F::One(), F(2)};
  std::vector<int32_t> rotations_;
  std::vector<CalculationInfo> calculations_;
  size_t num_intermediates_ = 0;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EVALUATOR_GRAPH_EVALUATOR_H_
