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
      Operands<std::unique_ptr<Expression<F>>, std::unique_ptr<Expression<F>>>
          operands) {
    return absl::WrapUnique(new SumExpression(std::move(operands)));
  }

  const Expression<F>* left() const { return operands_.left.get(); }
  const Expression<F>* right() const { return operands_.right.get(); }

  // Expression methods
  size_t Degree() const override {
    return std::max(operands_.left->Degree(), operands_.right->Degree());
  }

  uint64_t Complexity() const override {
    return operands_.left->Complexity() + operands_.right->Complexity() +
           kOverhead;
  }

  std::string ToString() const override {
    return absl::Substitute("{type: $0, (left, right): ($1, $2)}",
                            ExpressionTypeToString(this->type_),
                            operands_.left->ToString(),
                            operands_.right->ToString());
  }

 private:
  friend class ExpressionFactory<F>;

  explicit SumExpression(
      Operands<std::unique_ptr<Expression<F>>, std::unique_ptr<Expression<F>>>
          operands)
      : Expression<F>(ExpressionType::kSum), operands_(std::move(operands)) {}

  Operands<std::unique_ptr<Expression<F>>, std::unique_ptr<Expression<F>>>
      operands_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_SUM_EXPRESSION_H_
