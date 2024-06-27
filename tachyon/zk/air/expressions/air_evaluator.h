#ifndef TACHYON_ZK_AIR_EXPRESSIONS_AIR_EVALUATOR_H_
#define TACHYON_ZK_AIR_EXPRESSIONS_AIR_EVALUATOR_H_

#include "absl/types/span.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/zk/air/expressions/first_row_expression.h"
#include "tachyon/zk/air/expressions/last_row_expression.h"
#include "tachyon/zk/air/expressions/transition_expression.h"
#include "tachyon/zk/air/expressions/variable_expression.h"
#include "tachyon/zk/expressions/evaluator.h"
#include "tachyon/zk/expressions/expression.h"

namespace tachyon::zk::air {

template <typename F>
class AirEvaluator : public Evaluator<F, F> {
 public:
  using RowMajorMatrix = math::RowMajorMatrix<F>;
  using ConstMatrixBlock = Eigen::Block<const RowMajorMatrix>;

  F Evaluate(const Expression<F>* input) override {
    switch (input->type()) {
      case ExpressionType::kConstant:
        return input->ToConstant()->value();
      case ExpressionType::kSum: {
        const SumExpression<F>* sum = input->ToSum();
        return Evaluate(sum->left()) + Evaluate(sum->right());
      }
      case ExpressionType::kProduct: {
        const ProductExpression<F>* product = input->ToProduct();
        return Evaluate(product->left()) * Evaluate(product->right());
      }
      case ExpressionType::kNegated: {
        const NegatedExpression<F>* negated = input->ToNegated();
        return -Evaluate(negated->expr());
      }
      case ExpressionType::kScaled: {
        const ScaledExpression<F>* scaled = input->ToScaled();
        return Evaluate(scaled->expr()) * scaled->scale();
      }
      case ExpressionType::kFirstRow: {
        if (current_row_ == 0) {
          const FirstRowExpression<F>* first_row = input->ToFirstRow();
          return Evaluate(first_row->expr());
        }
        return F::Zero();
      }
      case ExpressionType::kLastRow: {
        if (current_row_ == num_rows_ - 2) {
          const LastRowExpression<F>* last_row = input->ToLastRow();
          return Evaluate(last_row->expr());
        }
        return F::Zero();
      }
      case ExpressionType::kTransition: {
        const TransitionExpression<F>* transition = input->ToTransition();
        return Evaluate(transition->expr());
      }
      case ExpressionType::kVariable: {
        Variable variable = input->ToVariable()->variable();
        switch (variable.type()) {
          case Variable::Type::kPublic:
            return public_values_[variable.row_index()];
          case Variable::Type::kMain:
            CHECK_NE(main_window_, nullptr);
            return main_window_->coeff(variable.row_index(),
                                       variable.col_index());
          case Variable::Type::kPreprocessed: {
            CHECK_NE(preprocessed_window_, nullptr);
            return preprocessed_window_->coeff(variable.row_index(),
                                               variable.col_index());
          }
          case Variable::Type::kChallenge:
          case Variable::Type::kPermutation:
            NOTIMPLEMENTED();
            return F::Zero();
        }
      }
      case ExpressionType::kAdvice:
      case ExpressionType::kChallenge:
      case ExpressionType::kFixed:
      case ExpressionType::kInstance:
      case ExpressionType::kSelector:
        NOTREACHED() << ExpressionTypeToString(input->type())
                     << " expression is not allowed in AIR!";
        return F::Zero();
    }
    NOTREACHED();
    return F::Zero();
  }

  void set_public_values(absl::Span<const F> public_values) {
    public_values_ = public_values;
  }

  void set_num_rows(Eigen::Index num_rows) { num_rows_ = num_rows; }

  void SetCurrentWindowData(Eigen::Index current_row,
                            const ConstMatrixBlock* main_window,
                            const ConstMatrixBlock* preprocessed_window) {
    current_row_ = current_row;
    main_window_ = main_window;
    preprocessed_window_ = preprocessed_window;
  }

 private:
  Eigen::Index num_rows_ = 0;
  absl::Span<const F> public_values_;

  Eigen::Index current_row_ = 0;

  // not owned
  const ConstMatrixBlock* main_window_ = nullptr;
  const ConstMatrixBlock* preprocessed_window_ = nullptr;
};

}  // namespace tachyon::zk::air

#endif  // TACHYON_ZK_AIR_EXPRESSIONS_AIR_EVALUATOR_H_
