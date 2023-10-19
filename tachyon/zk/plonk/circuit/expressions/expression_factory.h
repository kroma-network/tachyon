// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EXPRESSION_FACTORY_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EXPRESSION_FACTORY_H_

#include <memory>
#include <utility>

#include "absl/memory/memory.h"

#include "tachyon/zk/plonk/circuit/expressions/advice_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/challenge_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/constant_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/expression.h"
#include "tachyon/zk/plonk/circuit/expressions/fixed_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/instance_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/negated_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/product_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/scaled_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/selector_expression.h"
#include "tachyon/zk/plonk/circuit/expressions/sum_expression.h"

namespace tachyon::zk {

template <typename F>
class ExpressionFactory {
 public:
  using Expr = Expression<F>;

  static std::unique_ptr<Expr> Constant(const F& value) {
    return absl::WrapUnique(new ConstantExpression<F>(value));
  }

  static std::unique_ptr<Expr> Selector(const Selector& selector) {
    return absl::WrapUnique(new SelectorExpression<F>(selector));
  }

  static std::unique_ptr<Expr> Fixed(const FixedQuery& query) {
    return absl::WrapUnique(new FixedExpression<F>(query));
  }

  static std::unique_ptr<Expr> Advice(const AdviceQuery& query) {
    return absl::WrapUnique(new AdviceExpression<F>(query));
  }

  static std::unique_ptr<Expr> Instance(const InstanceQuery& query) {
    return absl::WrapUnique(new InstanceExpression<F>(query));
  }

  static std::unique_ptr<Expr> Challenge(const Challenge& challenge) {
    return absl::WrapUnique(new ChallengeExpression<F>(challenge));
  }

  static std::unique_ptr<Expr> Negated(std::unique_ptr<Expr> value) {
    return absl::WrapUnique(new NegatedExpression<F>(std::move(value)));
  }

  static std::unique_ptr<Expr> Sum(
      Operands<std::unique_ptr<Expr>, std::unique_ptr<Expr>> operands) {
    return absl::WrapUnique(new SumExpression<F>(std::move(operands)));
  }

  static std::unique_ptr<Expr> Product(
      Operands<std::unique_ptr<Expr>, std::unique_ptr<Expr>> operands) {
    return absl::WrapUnique(new ProductExpression<F>(std::move(operands)));
  }

  static std::unique_ptr<Expr> Scaled(
      Operands<std::unique_ptr<Expr>, F> operands) {
    return absl::WrapUnique(new ScaledExpression<F>(std::move(operands)));
  }
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXPRESSIONS_EXPRESSION_FACTORY_H_
