#ifndef TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_
#define TACHYON_ZK_AIR_CONSTRAINT_SYSTEM_CONSTRAINT_SYSTEM_H_

#include <stddef.h>

#include <memory>
#include <utility>
#include <vector>

#include "tachyon/math/matrix/matrix_types.h"
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

  bool IsSatisfied(AirEvaluator<F>& evaluator,
                   const std::vector<F>& public_values,
                   const RowMajorMatrix& main,
                   const RowMajorMatrix* preprocessed = nullptr) const {
    CheckInputDimensions(public_values, main, preprocessed);

    evaluator.set_public_values(public_values);
    evaluator.set_num_rows(main.rows());

    for (Eigen::Index i = 0; i < main.rows() - 1; ++i) {
      const ConstMatrixBlock main_window(main, i, 0, 2, main_width_);

      if (preprocessed) {
        new (preprocessed_buf_.get())
            ConstMatrixBlock(*preprocessed, i, 0, 2, preprocessed_width_);
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

 private:
  constexpr void CheckInputDimensions(
      const std::vector<F>& public_values, const RowMajorMatrix& main,
      const RowMajorMatrix* preprocessed = nullptr) const {
    CHECK_EQ(public_values.size(), num_public_values_);
    CHECK_EQ(main.cols(), main_width_);
    CHECK_GE(main.rows(), 1);
    if (preprocessed) {
      CHECK_EQ(preprocessed->cols(), preprocessed_width_);
      CHECK_EQ(preprocessed->rows(), main.rows());
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
