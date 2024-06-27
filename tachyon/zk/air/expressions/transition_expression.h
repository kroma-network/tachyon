#ifndef TACHYON_ZK_AIR_EXPRESSIONS_TRANSITION_EXPRESSION_H_
#define TACHYON_ZK_AIR_EXPRESSIONS_TRANSITION_EXPRESSION_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"

#include "tachyon/zk/expressions/expression.h"

namespace tachyon::zk::air {

template <typename F>
class ExpressionFactory;

template <typename F>
class TransitionExpression : public Expression<F> {
 public:
  static std::unique_ptr<TransitionExpression> CreateForTesting(
      std::unique_ptr<Expression<F>> expr) {
    return absl::WrapUnique(new TransitionExpression(std::move(expr)));
  }

  Expression<F>* expr() const { return expr_.get(); }

  // Expression methods
  size_t Degree() const override { return expr_->Degree(); }

  uint64_t Complexity() const override { return expr_->Complexity(); }

  std::unique_ptr<Expression<F>> Clone() const override {
    return absl::WrapUnique(new TransitionExpression(expr_->Clone()));
  }

  std::string ToString() const override {
    return absl::Substitute("{type: $0, expr: $1}",
                            ExpressionTypeToString(this->type_),
                            expr_->ToString());
  }

  bool operator==(const Expression<F>& other) const override {
    if (!Expression<F>::operator==(other)) return false;
    const TransitionExpression* transition = other.ToTransition();
    return *expr_ == *transition->expr_;
  }

 private:
  friend class ExpressionFactory<F>;

  explicit TransitionExpression(std::unique_ptr<Expression<F>> expr)
      : Expression<F>(ExpressionType::kTransition), expr_(std::move(expr)) {}

  std::unique_ptr<Expression<F>> expr_;
};

}  // namespace tachyon::zk::air

#endif  // TACHYON_ZK_AIR_EXPRESSIONS_TRANSITION_EXPRESSION_H_
