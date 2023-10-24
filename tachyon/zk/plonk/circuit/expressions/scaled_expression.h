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
      std::unique_ptr<Expression<F>> expr, const F& scale) {
    return absl::WrapUnique(new ScaledExpression(std::move(expr), scale));
  }

  const Expression<F>* expr() const { return expr_.get(); }
  const F& scale() const { return scale_; }

  bool operator==(const Expression<F>& other) const {
    if (!Expression<F>::operator==(other)) return false;
    const ScaledExpression* scaled = other.ToScaled();
    return *expr_ == *scaled->expr_ && scale_ == scaled->scale;
  }
  bool operator!=(const Expression<F>& other) const {
    return !operator==(other);
  }

  // Expression methods
  size_t Degree() const override { return expr_->Degree(); }

  uint64_t Complexity() const override {
    return expr_->Complexity() + kOverhead;
  }

  std::unique_ptr<Expression<F>> Clone() const override {
    return absl::WrapUnique(new ScaledExpression(expr_->Clone(), scale_));
  }

  std::string ToString() const override {
    return absl::Substitute("{type: $0, expr: $1, scale: $2}",
                            ExpressionTypeToString(this->type_),
                            expr_->ToString(), scale_.ToString());
  }

 private:
  friend class ExpressionFactory<F>;

  ScaledExpression(std::unique_ptr<Expression<F>> expr, const F& scale)
      : Expression<F>(ExpressionType::kScaled),
        expr_(std::move(expr)),
        scale_(scale) {}
  ScaledExpression(std::unique_ptr<Expression<F>> expr, F&& scale)
      : Expression<F>(ExpressionType::kScaled),
        expr_(std::move(expr)),
        scale_(std::move(scale)) {}

  std::unique_ptr<Expression<F>> expr_;
  F scale_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_SCALED_EXPRESSION_H_
