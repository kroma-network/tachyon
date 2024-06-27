// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_EXPRESSIONS_EXPRESSION_FACTORY_H_
#define TACHYON_ZK_EXPRESSIONS_EXPRESSION_FACTORY_H_

#include <memory>
#include <utility>

#include "absl/memory/memory.h"

#include "tachyon/zk/expressions/advice_expression.h"
#include "tachyon/zk/expressions/challenge_expression.h"
#include "tachyon/zk/expressions/constant_expression.h"
#include "tachyon/zk/expressions/expression.h"
#include "tachyon/zk/expressions/fixed_expression.h"
#include "tachyon/zk/expressions/instance_expression.h"
#include "tachyon/zk/expressions/negated_expression.h"
#include "tachyon/zk/expressions/product_expression.h"
#include "tachyon/zk/expressions/scaled_expression.h"
#include "tachyon/zk/expressions/selector_expression.h"
#include "tachyon/zk/expressions/sum_expression.h"

namespace tachyon::zk {

template <typename F>
class ExpressionFactory {
 public:
  using Expr = Expression<F>;

  static std::unique_ptr<Expr> Constant(const F& value) {
    return absl::WrapUnique(new ConstantExpression<F>(value));
  }

  static std::unique_ptr<Expr> Selector(plonk::Selector selector) {
    return absl::WrapUnique(new SelectorExpression<F>(selector));
  }

  static std::unique_ptr<Expr> Fixed(const plonk::FixedQuery& query) {
    return absl::WrapUnique(new FixedExpression<F>(query));
  }

  static std::unique_ptr<Expr> Advice(const plonk::AdviceQuery& query) {
    return absl::WrapUnique(new AdviceExpression<F>(query));
  }

  static std::unique_ptr<Expr> Instance(const plonk::InstanceQuery& query) {
    return absl::WrapUnique(new InstanceExpression<F>(query));
  }

  static std::unique_ptr<Expr> Challenge(plonk::Challenge challenge) {
    return absl::WrapUnique(new ChallengeExpression<F>(challenge));
  }

  static std::unique_ptr<Expr> Negated(std::unique_ptr<Expr> value) {
    return absl::WrapUnique(new NegatedExpression<F>(std::move(value)));
  }

  static std::unique_ptr<Expr> Sum(std::unique_ptr<Expr> left,
                                   std::unique_ptr<Expr> right) {
    return absl::WrapUnique(
        new SumExpression<F>(std::move(left), std::move(right)));
  }

  static std::unique_ptr<Expr> Product(std::unique_ptr<Expr> left,
                                       std::unique_ptr<Expr> right) {
    return absl::WrapUnique(
        new ProductExpression<F>(std::move(left), std::move(right)));
  }

  static std::unique_ptr<Expr> Scaled(std::unique_ptr<Expr> left,
                                      const F& scale) {
    return absl::WrapUnique(new ScaledExpression<F>(std::move(left), scale));
  }
};

template <typename F>
std::unique_ptr<Expression<F>> operator+(
    const std::unique_ptr<Expression<F>>& lhs,
    const std::unique_ptr<Expression<F>>& rhs) {
  return operator+(lhs->Clone(), rhs->Clone());
}

template <typename F>
std::unique_ptr<Expression<F>> operator+(std::unique_ptr<Expression<F>>&& lhs,
                                         std::unique_ptr<Expression<F>>&& rhs) {
  return ExpressionFactory<F>::Sum(std::move(lhs), std::move(rhs));
}

template <typename F>
std::unique_ptr<Expression<F>> operator-(
    const std::unique_ptr<Expression<F>>& lhs,
    const std::unique_ptr<Expression<F>>& rhs) {
  return operator-(lhs->Clone(), rhs->Clone());
}

template <typename F>
std::unique_ptr<Expression<F>> operator-(std::unique_ptr<Expression<F>>&& lhs,
                                         std::unique_ptr<Expression<F>>&& rhs) {
  return ExpressionFactory<F>::Sum(std::move(lhs), operator-(std::move(rhs)));
}

template <typename F>
std::unique_ptr<Expression<F>> operator*(
    const std::unique_ptr<Expression<F>>& lhs,
    const std::unique_ptr<Expression<F>>& rhs) {
  return operator*(lhs->Clone(), rhs->Clone());
}

template <typename F>
std::unique_ptr<Expression<F>> operator*(std::unique_ptr<Expression<F>>&& lhs,
                                         std::unique_ptr<Expression<F>>&& rhs) {
  return ExpressionFactory<F>::Product(std::move(lhs), std::move(rhs));
}

template <typename F>
std::unique_ptr<Expression<F>> operator*(
    const std::unique_ptr<Expression<F>>& expr, const F& scale) {
  return operator*(expr->Clone(), scale);
}

template <typename F>
std::unique_ptr<Expression<F>> operator*(std::unique_ptr<Expression<F>>&& expr,
                                         const F& scale) {
  return ExpressionFactory<F>::Scaled(std::move(expr), scale);
}

template <typename F>
std::unique_ptr<Expression<F>> operator-(
    const std::unique_ptr<Expression<F>>& expr) {
  return operator-(expr->Clone());
}

template <typename F>
std::unique_ptr<Expression<F>> operator-(
    std::unique_ptr<Expression<F>>&& expr) {
  return ExpressionFactory<F>::Negated(std::move(expr));
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_EXPRESSIONS_EXPRESSION_FACTORY_H_
