// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_EXPRESSIONS_NEGATED_EXPRESSION_H_
#define TACHYON_ZK_EXPRESSIONS_NEGATED_EXPRESSION_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"

#include "tachyon/zk/expressions/expression.h"

namespace tachyon::zk {

template <typename F>
class NegatedExpression : public Expression<F> {
 public:
  constexpr static uint64_t kOverhead = 30;

  static std::unique_ptr<NegatedExpression> CreateForTesting(
      std::unique_ptr<Expression<F>> expr) {
    return absl::WrapUnique(new NegatedExpression(std::move(expr)));
  }

  Expression<F>* expr() const { return expr_.get(); }

  // Expression methods
  size_t Degree() const override { return expr_->Degree(); }

  uint64_t Complexity() const override {
    return expr_->Complexity() + kOverhead;
  }

  std::unique_ptr<Expression<F>> Clone() const override {
    return absl::WrapUnique(new NegatedExpression(expr_->Clone()));
  }

  std::string ToString() const override {
    return absl::Substitute("{type: $0, expr: $1}",
                            ExpressionTypeToString(this->type_),
                            expr_->ToString());
  }

  void WriteIdentifier(std::ostream& out) const override {
    out << "(-";
    expr_->WriteIdentifier(out);
    out << ")";
  }

  bool operator==(const Expression<F>& other) const override {
    if (!Expression<F>::operator==(other)) return false;
    const NegatedExpression* negated = other.ToNegated();
    return *expr_ == *negated->expr_;
  }

 private:
  friend class ExpressionFactory<F>;

  explicit NegatedExpression(std::unique_ptr<Expression<F>> expr)
      : Expression<F>(ExpressionType::kNegated), expr_(std::move(expr)) {}

  std::unique_ptr<Expression<F>> expr_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_EXPRESSIONS_NEGATED_EXPRESSION_H_
