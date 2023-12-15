#ifndef TACHYON_ZK_PLONK_CIRCUIT_CONSTRAINT_H_
#define TACHYON_ZK_PLONK_CIRCUIT_CONSTRAINT_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "tachyon/zk/expressions/expression.h"

namespace tachyon::zk {

// An individual polynomial constraint.
//
// These are returned by the closures passed to
// |ConstraintSystem::CreateGate()|.
template <typename F>
class Constraint {
 public:
  explicit Constraint(std::unique_ptr<Expression<F>> expression)
      : expression_(std::move(expression)) {}
  Constraint(std::string_view name, std::unique_ptr<Expression<F>> expression)
      : name_(std::string(name)), expression_(std::move(expression)) {}

  const std::string& name() const& { return name_; }
  const std::unique_ptr<Expression<F>>& expression() const& {
    return expression_;
  }

  std::string&& TakeName() && { return std::move(name_); }
  std::unique_ptr<Expression<F>>&& TakeExpression() && {
    return std::move(expression_);
  }

  std::string ToString() const {
    return absl::Substitute("{name: $0, expression: $1}", name_,
                            expression_->ToString());
  }

 private:
  std::string name_;
  std::unique_ptr<Expression<F>> expression_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_CONSTRAINT_H_
