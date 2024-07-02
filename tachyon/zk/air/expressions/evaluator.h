#ifndef TACHYON_ZK_AIR_EXPRESSIONS_EVALUATOR_H_
#define TACHYON_ZK_AIR_EXPRESSIONS_EVALUATOR_H_

#include <optional>
#include <vector>

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/base/logging.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/zk/air/expressions/first_row_expression.h"
#include "tachyon/zk/air/expressions/last_row_expression.h"
#include "tachyon/zk/air/expressions/transition_expression.h"
#include "tachyon/zk/air/expressions/variable_expression.h"
#include "tachyon/zk/expressions/expression.h"

namespace tachyon::zk::air {

template <typename F>
class Evaluator {
 public:
  F Evaluate(const Expression<F>* input, Eigen::Index current_row,
             Eigen::Index num_rows, const std::vector<F>& public_values,
             std::optional<const Eigen::Ref<const math::RowMajorMatrix<F>>>
                 preprocessed_window,
             const math::RowMajorMatrix<F>& main_window) {
    switch (input->type()) {
      case ExpressionType::kConstant:
        return input->ToConstant()->value();
      case ExpressionType::kSum: {
        const SumExpression<F>* sum = input->ToSum();
        return Evaluate(sum->left(), current_row, num_rows, public_values,
                        preprocessed_window, main_window) +
               Evaluate(sum->right(), current_row, num_rows, public_values,
                        preprocessed_window, main_window);
      }
      case ExpressionType::kProduct: {
        const ProductExpression<F>* product = input->ToProduct();
        return Evaluate(product->left(), current_row, num_rows, public_values,
                        preprocessed_window, main_window) *
               Evaluate(product->right(), current_row, num_rows, public_values,
                        preprocessed_window, main_window);
      }
      case ExpressionType::kNegated: {
        const NegatedExpression<F>* negated = input->ToNegated();
        return -Evaluate(negated->expr(), current_row, num_rows, public_values,
                         preprocessed_window, main_window);
      }
      case ExpressionType::kScaled: {
        const ScaledExpression<F>* scaled = input->ToScaled();
        return Evaluate(scaled->expr(), current_row, num_rows, public_values,
                        preprocessed_window, main_window) *
               scaled->scale();
      }
      case ExpressionType::kFirstRow: {
        if (current_row == 0) {
          const FirstRowExpression<F>* first_row = input->ToFirstRow();
          return Evaluate(first_row->expr(), current_row, num_rows,
                          public_values, preprocessed_window, main_window);
        }
        return F::Zero();
      }
      case ExpressionType::kLastRow: {
        if (current_row == num_rows - 2) {
          const LastRowExpression<F>* last_row = input->ToLastRow();
          return Evaluate(last_row->expr(), current_row, num_rows,
                          public_values, preprocessed_window, main_window);
        }
        return F::Zero();
      }
      case ExpressionType::kTransition: {
        const TransitionExpression<F>* transition = input->ToTransition();
        return Evaluate(transition->expr(), current_row, num_rows,
                        public_values, preprocessed_window, main_window);
      }
      case ExpressionType::kVariable: {
        Variable variable = input->ToVariable()->variable();
        switch (variable.type()) {
          case Variable::Type::kPublic:
            return public_values[variable.row_index()];
          case Variable::Type::kMain:
            return main_window.coeff(variable.row_index(),
                                     variable.col_index());
          case Variable::Type::kPreprocessed: {
            if (preprocessed_window.has_value()) {
              auto prep = preprocessed_window.value();
              return prep.coeff(variable.row_index(), variable.col_index());
            } else {
              NOTREACHED();
            }
          }
          case Variable::Type::kChallenge:
          case Variable::Type::kPermutation:
            NOTREACHED() << "unimplemented!";
        }
      }
      case ExpressionType::kAdvice:
      case ExpressionType::kChallenge:
      case ExpressionType::kFixed:
      case ExpressionType::kInstance:
      case ExpressionType::kSelector:
        NOTREACHED() << ExpressionTypeToString(input->type())
                     << " expression is not allowed in AIR!";
    }
    NOTREACHED();
    return F::Zero();
  }
};

}  // namespace tachyon::zk::air

#endif  // TACHYON_ZK_AIR_EXPRESSIONS_EVALUATOR_H_
