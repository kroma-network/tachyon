// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_EXPRESSIONS_EXPRESSION_H_
#define TACHYON_ZK_EXPRESSIONS_EXPRESSION_H_

#include <memory>
#include <string>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/zk/expressions/expression_type.h"

namespace tachyon::zk {

template <typename F>
class ExpressionFactory;

template <typename F, typename T>
class Evaluator;

template <typename F>
class ConstantExpression;

template <typename F>
class NegatedExpression;

template <typename F>
class SumExpression;

template <typename F>
class ProductExpression;

template <typename F>
class ScaledExpression;

namespace plonk {
template <typename F>
class SelectorExpression;

template <typename F>
class FixedExpression;

template <typename F>
class AdviceExpression;

template <typename F>
class InstanceExpression;

template <typename F>
class ChallengeExpression;
}  // namespace plonk

namespace air {
template <typename F>
class FirstRowExpression;

template <typename F>
class LastRowExpression;

template <typename F>
class TransitionExpression;

template <typename F>
class VariableExpression;
}  // namespace air

// A Expression represents a polynomial.
template <typename F>
class Expression {
 public:
  virtual ~Expression() = default;

  ExpressionType type() const { return type_; }

  // Returns the degree of the polynomial
  virtual size_t Degree() const = 0;

  // Returns the approximated computational complexity of this expression.
  virtual uint64_t Complexity() const = 0;
  virtual std::string ToString() const = 0;

  virtual std::unique_ptr<Expression> Clone() const = 0;

  virtual bool operator==(const Expression& other) const {
    return type_ == other.type_;
  }
  bool operator!=(const Expression& other) const { return !operator==(other); }

  std::unique_ptr<Expression> operator-() const {
    return ExpressionFactory<F>::Negated(Clone());
  }

  static std::vector<std::unique_ptr<Expression>> CloneExpressions(
      const std::vector<std::unique_ptr<Expression>>& expressions) {
    return base::CreateVector(expressions.size(), [&expressions](size_t i) {
      return expressions[i]->Clone();
    });
  }

  template <typename Evaluated>
  Evaluated Evaluate(Evaluator<F, Evaluated>* evaluator) const {
    return evaluator->Evaluate(this);
  }

#define DEFINE_CONVERSION_METHOD(type)                    \
  type##Expression<F>* To##type() {                       \
    CHECK_EQ(type_, ExpressionType::k##type);             \
    return static_cast<type##Expression<F>*>(this);       \
  }                                                       \
                                                          \
  const type##Expression<F>* To##type() const {           \
    CHECK_EQ(type_, ExpressionType::k##type);             \
    return static_cast<const type##Expression<F>*>(this); \
  }

  DEFINE_CONVERSION_METHOD(Constant)
  DEFINE_CONVERSION_METHOD(Negated)
  DEFINE_CONVERSION_METHOD(Sum)
  DEFINE_CONVERSION_METHOD(Product)
  DEFINE_CONVERSION_METHOD(Scaled)

#undef DEFINE_CONVERSION_METHOD

#define DEFINE_NAMESPACED_CONVERSION_METHOD(ns, type)         \
  ns::type##Expression<F>* To##type() {                       \
    CHECK_EQ(type_, ExpressionType::k##type);                 \
    return static_cast<ns::type##Expression<F>*>(this);       \
  }                                                           \
                                                              \
  const ns::type##Expression<F>* To##type() const {           \
    CHECK_EQ(type_, ExpressionType::k##type);                 \
    return static_cast<const ns::type##Expression<F>*>(this); \
  }
  DEFINE_NAMESPACED_CONVERSION_METHOD(plonk, Selector)
  DEFINE_NAMESPACED_CONVERSION_METHOD(plonk, Fixed)
  DEFINE_NAMESPACED_CONVERSION_METHOD(plonk, Advice)
  DEFINE_NAMESPACED_CONVERSION_METHOD(plonk, Instance)
  DEFINE_NAMESPACED_CONVERSION_METHOD(plonk, Challenge)
  DEFINE_NAMESPACED_CONVERSION_METHOD(air, FirstRow)
  DEFINE_NAMESPACED_CONVERSION_METHOD(air, LastRow)
  DEFINE_NAMESPACED_CONVERSION_METHOD(air, Transition)
  DEFINE_NAMESPACED_CONVERSION_METHOD(air, Variable)
#undef DEFINE_NAMESPACED_CONVERSION_METHOD

 protected:
  explicit Expression(ExpressionType type) : type_(type) {}

  ExpressionType type_;
};

template <typename F>
std::ostream& operator<<(std::ostream& os, const Expression<F>& expression) {
  return os << expression.ToString();
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_EXPRESSIONS_EXPRESSION_H_
