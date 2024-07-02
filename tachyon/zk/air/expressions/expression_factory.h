#ifndef TACHYON_ZK_AIR_EXPRESSIONS_EXPRESSION_FACTORY_H_
#define TACHYON_ZK_AIR_EXPRESSIONS_EXPRESSION_FACTORY_H_

#include <memory>
#include <utility>

#include "absl/memory/memory.h"

#include "tachyon/zk/air/expressions/first_row_expression.h"
#include "tachyon/zk/air/expressions/last_row_expression.h"
#include "tachyon/zk/air/expressions/transition_expression.h"
#include "tachyon/zk/air/expressions/variable_expression.h"
#include "tachyon/zk/expressions/expression_factory.h"

namespace tachyon::zk::air {

template <typename F>
class ExpressionFactory : public tachyon::zk::ExpressionFactory<F> {
 public:
  using Expr = Expression<F>;

  static std::unique_ptr<Expr> FirstRow(std::unique_ptr<Expr> expr) {
    return absl::WrapUnique(new FirstRowExpression<F>(std::move(expr)));
  }

  static std::unique_ptr<Expr> LastRow(std::unique_ptr<Expr> expr) {
    return absl::WrapUnique(new LastRowExpression<F>(std::move(expr)));
  }

  static std::unique_ptr<Expr> Transition(std::unique_ptr<Expr> expr) {
    return absl::WrapUnique(new TransitionExpression<F>(std::move(expr)));
  }

  static std::unique_ptr<Expr> Variable(const Variable& variable) {
    return absl::WrapUnique(new VariableExpression<F>(variable));
  }
};

}  // namespace tachyon::zk::air

#endif  // TACHYON_ZK_AIR_EXPRESSIONS_EXPRESSION_FACTORY_H_
