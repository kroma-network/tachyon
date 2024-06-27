#ifndef TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_
#define TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_

#include <stddef.h>

#include <memory>
#include <utility>
#include <vector>

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/zk/air/constraint_system/variable.h"
#include "tachyon/zk/air/expressions/evaluator.h"

namespace tachyon::zk::air {

template <typename F>
class ConstraintSystem {
 public:
  ConstraintSystem() = default;
  ConstraintSystem(size_t preprocessed_width, size_t main_width,
                   size_t num_public_values)
      : preprocessed_width_(preprocessed_width),
        main_width_(main_width),
        num_public_values_(num_public_values) {
    preprocessed_.reserve(preprocessed_width * 2);
    main_.reserve(main_width * 2);
    public_values_.reserve(num_public_values);

    for (size_t row : {0, 1}) {
      for (Eigen::Index col = 0; col < preprocessed_width_; ++col) {
        preprocessed_.push_back(Variable::Preprocessed(row, col));
      }
    }
    for (size_t row : {0, 1}) {
      for (Eigen::Index col = 0; col < main_width_; ++col) {
        main_.push_back(Variable::Main(row, col));
      }
    }
    for (size_t i = 0; i < num_public_values; ++i) {
      public_values_.push_back(Variable::Public(i));
    }
  }

  const std::vector<Variable>& public_values() const { return public_values_; }

  const std::vector<Variable>& preprocessed() const { return preprocessed_; }

  const std::vector<Variable>& main() const { return main_; }

  void EnforceFirstRowConstraint(std::unique_ptr<Expression<F>> constraint) {
    EnforceConstraint(ExpressionFactory<F>::FirstRow(std::move(constraint)));
  }

  void EnforceLastRowConstraint(std::unique_ptr<Expression<F>> constraint) {
    EnforceConstraint(ExpressionFactory<F>::LastRow(std::move(constraint)));
  }

  void EnforceTransitionConstraint(std::unique_ptr<Expression<F>> constraint) {
    EnforceConstraint(ExpressionFactory<F>::Transition(std::move(constraint)));
  }

  void EnforceConstraint(std::unique_ptr<Expression<F>> constraint) {
    constraints_.push_back(std::move(constraint));
  }

  bool IsSatisfied(Evaluator<F>& evaluator, const std::vector<F>& public_values,
                   const Eigen::Ref<const math::RowMajorMatrix<F>> main) const {
    CHECK(public_values.size() == num_public_values_);
    CHECK(main.cols() == main_width_);

    for (Eigen::Index i = 0; i < main.rows() - 1; ++i) {
      for (const auto& constraint : constraints_) {
        if (!evaluator
                 .Evaluate(constraint.get(), i, main.rows(), public_values,
                           std::nullopt, main.middleRows(i, 2))
                 .IsZero()) {
          NOTREACHED() << i << constraint->ToString();
          return false;
        }
      }
    }
    return true;
  }

  bool IsSatisfied(Evaluator<F>& evaluator, const std::vector<F>& public_values,
                   const math::RowMajorMatrix<F>& preprocessed,
                   const math::RowMajorMatrix<F>& main) const {
    CHECK(public_values.size() == num_public_values_);
    CHECK(main.cols() == main_width_);
    CHECK(preprocessed.cols() == preprocessed_width_);
    CHECK(preprocessed.rows() == main.rows());

    for (Eigen::Index i = 0; i < main.rows() - 1; ++i) {
      for (const auto& constraint : constraints_) {
        if (!evaluator
                 .Evaluate(constraint.get(), i, main.rows(), public_values,
                           preprocessed.middleRows(i, 2), main.middleRows(i, 2))
                 .IsZero()) {
          NOTREACHED() << i << constraint->ToString();
          return false;
        }
      }
    }
    return true;
  }

 private:
  Eigen::Index preprocessed_width_ = 0;
  Eigen::Index main_width_ = 2;
  size_t num_public_values_ = 0;
  std::vector<Variable> preprocessed_;
  std::vector<Variable> main_;
  std::vector<Variable> public_values_;
  std::vector<std::unique_ptr<Expression<F>>> constraints_;
};

}  // namespace tachyon::zk::air

#endif  // TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_
