// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_SCALED_EXPRESSION_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_SCALED_EXPRESSION_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"

#include "tachyon/zk/plonk/circuit/expressions/expression.h"

namespace tachyon::zk {

template <typename F>
class ScaledExpression : public Expression<F> {
 public:
  constexpr static uint64_t kOverhead = 30;

  static std::unique_ptr<ScaledExpression> CreateForTesting(
      Operands<std::unique_ptr<Expression<F>>, F> operands) {
    return absl::WrapUnique(new ScaledExpression(std::move(operands)));
  }

  const Expression<F>* left() const { return operands_.left.get(); }
  const F& right() const { return operands_.right; }

  // Expression methods
  size_t Degree() const override { return operands_.left->Degree(); }

  uint64_t Complexity() const override {
    return operands_.left->Complexity() + kOverhead;
  }

  std::string ToString() const override {
    return absl::Substitute(
        "{type: $0, poly: $1, scalar: $2}", ExpressionTypeToString(this->type_),
        operands_.left->ToString(), operands_.right.ToString());
  }

 private:
  friend class ExpressionFactory<F>;

  explicit ScaledExpression(
      Operands<std::unique_ptr<Expression<F>>, F> operands)
      : Expression<F>(ExpressionType::kScaled),
        operands_(std::move(operands)) {}

  Operands<std::unique_ptr<Expression<F>>, F> operands_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_SCALED_EXPRESSION_H_
