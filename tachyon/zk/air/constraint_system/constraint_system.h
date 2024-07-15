#ifndef TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_
#define TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/zk/air/constraint_system/trace.h"
#include "tachyon/zk/air/constraint_system/variable.h"
#include "tachyon/zk/air/expressions/air_evaluator.h"

namespace tachyon::zk::air {

template <typename F>
class ConstraintSystem {
 public:
  using RowMajorMatrix = math::RowMajorMatrix<F>;
  using ConstMatrixBlock = Eigen::Block<const RowMajorMatrix>;

  ConstraintSystem() = default;
  ConstraintSystem(size_t num_public_values, size_t main_width,
                   size_t preprocessed_width)
      : num_public_values_(num_public_values),
        main_width_(main_width),
        preprocessed_width_(preprocessed_width),
        preprocessed_buf_(std::make_unique<char[]>(sizeof(ConstMatrixBlock))) {}

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

  bool IsSatisfied(AirEvaluator<F, F>& evaluator,
                   const std::vector<F>& public_values,
                   const Trace<F>& trace) const {
    CheckInputDimensions(public_values, trace);

    evaluator.set_public_values(public_values);
    evaluator.set_num_rows(trace.rows());

    for (Eigen::Index i = 0; i < trace.rows() - 1; ++i) {
      const ConstMatrixBlock main_window(trace.main, i, 0, 2, main_width_);

      if (trace.preprocessed) {
        new (preprocessed_buf_.get())
            ConstMatrixBlock(*trace.preprocessed, i, 0, 2, preprocessed_width_);
        evaluator.SetCurrentWindowData(
            i, &main_window,
            reinterpret_cast<const ConstMatrixBlock*>(preprocessed_buf_.get()));
      } else {
        evaluator.SetCurrentWindowData(i, &main_window, nullptr);
      }

      for (const std::unique_ptr<Expression<F>>& constraint : constraints_) {
        if (!evaluator.Evaluate(constraint.get()).IsZero()) {
          return false;
        }
      }
    }
    return true;
  }

  size_t GetMaxConstraintDegree() const {
    size_t degree = 0;
    for (const std::unique_ptr<Expression<F>>& constraint : constraints_) {
      degree = std::max(degree, constraint->Degree());
    }
    return degree;
  }

 private:
  constexpr void CheckInputDimensions(const std::vector<F>& public_values,
                                      const Trace<F>& trace) const {
    CHECK_EQ(public_values.size(), num_public_values_);
    CHECK_EQ(trace.main.cols(), main_width_);
    CHECK_GE(trace.rows(), 1);
    if (trace.preprocessed) {
      CHECK_EQ(trace.preprocessed->cols(), preprocessed_width_);
    } else {
      CHECK_EQ(preprocessed_width_, 0);
    }
  }

  const size_t num_public_values_ = 0;
  const Eigen::Index main_width_ = 0;
  const Eigen::Index preprocessed_width_ = 0;
  std::vector<std::unique_ptr<Expression<F>>> constraints_;
  std::unique_ptr<char[]> preprocessed_buf_;
};

}  // namespace tachyon::zk::air

#endif  // TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_
