// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_SUM_EXPRESSION_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_SUM_EXPRESSION_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"

#include "tachyon/zk/plonk/circuit/expressions/expression.h"

namespace tachyon::zk {

template <typename F>
class SumExpression : public Expression<F> {
 public:
  constexpr static uint64_t kOverhead = 15;
  static std::unique_ptr<SumExpression> CreateForTesting(
      std::unique_ptr<Expression<F>> left,
      std::unique_ptr<Expression<F>> right) {
    return absl::WrapUnique(
        new SumExpression(std::move(left), std::move(right)));
  }

  const Expression<F>* left() const { return left_.get(); }
  const Expression<F>* right() const { return right_.get(); }

  bool operator==(const Expression<F>& other) const {
    if (!Expression<F>::operator==(other)) return false;
    const SumExpression* sum = other.ToSum();
    return *left_ == *sum->left_ && *right_ == *sum->right_;
  }
  bool operator!=(const Expression<F>& other) const {
    return !operator==(other);
  }

  // Expression methods
  size_t Degree() const override {
    return std::max(left_->Degree(), right_->Degree());
  }

  uint64_t Complexity() const override {
    return left_->Complexity() + right_->Complexity() + kOverhead;
  }

  std::string ToString() const override {
    return absl::Substitute("{type: $0, left: $1 right: $2}",
                            ExpressionTypeToString(this->type_),
                            left_->ToString(), right_->ToString());
  }

 private:
  friend class ExpressionFactory<F>;

  SumExpression(std::unique_ptr<Expression<F>> left,
                std::unique_ptr<Expression<F>> right)
      : Expression<F>(ExpressionType::kSum),
        left_(std::move(left)),
        right_(std::move(right)) {}

  std::unique_ptr<Expression<F>> left_;
  std::unique_ptr<Expression<F>> right_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_SUM_EXPRESSION_H_
