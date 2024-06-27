#ifndef TACHYON_ZK_AIR_EXPRESSIONS_VARIABLE_EXPRESSION_H_
#define TACHYON_ZK_AIR_EXPRESSIONS_VARIABLE_EXPRESSION_H_

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"

#include "tachyon/zk/air/constraint_system/variable.h"
#include "tachyon/zk/expressions/expression.h"

namespace tachyon::zk::air {

template <typename F>
class ExpressionFactory;

template <typename F>
class VariableExpression : public Expression<F> {
 public:
  static std::unique_ptr<VariableExpression> CreateForTesting(
      const Variable& variable) {
    return absl::WrapUnique(new VariableExpression(variable));
  }

  const Variable& variable() const { return variable_; }

  // Expression methods
  size_t Degree() const override { return variable_.Degree(); }

  uint64_t Complexity() const override { return 1; }

  std::unique_ptr<Expression<F>> Clone() const override {
    return absl::WrapUnique(new VariableExpression(variable_));
  }

  std::string ToString() const override {
    return absl::Substitute("{type: $0, variable: $1}",
                            ExpressionTypeToString(this->type_),
                            variable_.ToString());
  }

  bool operator==(const Expression<F>& other) const override {
    if (!Expression<F>::operator==(other)) return false;
    const VariableExpression* variable = other.ToVariable();
    return variable_ == variable->variable();
  }

 private:
  friend class ExpressionFactory<F>;

  explicit VariableExpression(const Variable& variable)
      : Expression<F>(ExpressionType::kVariable), variable_(variable) {}

  Variable variable_;
};

}  // namespace tachyon::zk::air

#endif  // TACHYON_ZK_AIR_EXPRESSIONS_VARIABLE_EXPRESSION_H_
