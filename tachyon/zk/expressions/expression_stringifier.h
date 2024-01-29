#ifndef TACHYON_ZK_EXPRESSIONS_EXPRESSION_STRINGIFIER_H_
#define TACHYON_ZK_EXPRESSIONS_EXPRESSION_STRINGIFIER_H_

#include <memory>
#include <string>
#include <vector>

#include "tachyon/base/logging.h"
#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/base/field_stringifier.h"
#include "tachyon/zk/expressions/advice_expression.h"
#include "tachyon/zk/expressions/challenge_expression.h"
#include "tachyon/zk/expressions/constant_expression.h"
#include "tachyon/zk/expressions/fixed_expression.h"
#include "tachyon/zk/expressions/instance_expression.h"
#include "tachyon/zk/expressions/negated_expression.h"
#include "tachyon/zk/expressions/product_expression.h"
#include "tachyon/zk/expressions/scaled_expression.h"
#include "tachyon/zk/expressions/selector_expression.h"
#include "tachyon/zk/expressions/sum_expression.h"
#include "tachyon/zk/plonk/base/phase_stringifier.h"
#include "tachyon/zk/plonk/constraint_system/challenge_stringifier.h"
#include "tachyon/zk/plonk/constraint_system/rotation_stringifier.h"

namespace tachyon::base::internal {

template <typename F>
class RustDebugStringifier<zk::Expression<F>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      const zk::Expression<F>& expression) {
    switch (expression.type()) {
      case zk::ExpressionType::kConstant: {
        return os << fmt.DebugTuple("Constant")
                         .Field(expression.ToConstant()->value())
                         .Finish();
      }
      case zk::ExpressionType::kSelector: {
        const zk::Selector& selector = expression.ToSelector()->selector();
        return os
               << fmt.DebugTuple("Selector").Field(selector.index()).Finish();
      }
      case zk::ExpressionType::kFixed: {
        const zk::FixedQuery& query = expression.ToFixed()->query();
        return os << fmt.DebugStruct("Fixed")
                         .Field("query_index", query.index())
                         .Field("column_index", query.column().index())
                         .Field("rotation", query.rotation())
                         .Finish();
      }
      case zk::ExpressionType::kAdvice: {
        const zk::AdviceQuery& query = expression.ToAdvice()->query();
        base::internal::DebugStruct debug_struct = fmt.DebugStruct("Advice");
        debug_struct.Field("query_index", query.index())
            .Field("column_index", query.column().index())
            .Field("rotation", query.rotation());
        if (query.column().phase() != zk::kFirstPhase) {
          debug_struct.Field("phase", query.column().phase());
        }
        return os << debug_struct.Finish();
      }
      case zk::ExpressionType::kInstance: {
        const zk::InstanceQuery& query = expression.ToInstance()->query();
        return os << fmt.DebugStruct("Instance")
                         .Field("query_index", query.index())
                         .Field("column_index", query.column().index())
                         .Field("rotation", query.rotation())
                         .Finish();
      }
      case zk::ExpressionType::kChallenge: {
        const zk::Challenge& challenge = expression.ToChallenge()->challenge();
        return os << fmt.DebugTuple("Challenge").Field(challenge).Finish();
      }
      case zk::ExpressionType::kNegated: {
        return os << fmt.DebugTuple("Negated")
                         .Field(*expression.ToNegated()->expr())
                         .Finish();
      }
      case zk::ExpressionType::kSum: {
        const zk::SumExpression<F>* sum = expression.ToSum();
        return os << fmt.DebugTuple("Sum")
                         .Field(*sum->left())
                         .Field(*sum->right())
                         .Finish();
      }
      case zk::ExpressionType::kProduct: {
        const zk::ProductExpression<F>* product = expression.ToProduct();
        return os << fmt.DebugTuple("Product")
                         .Field(*product->left())
                         .Field(*product->right())
                         .Finish();
      }
      case zk::ExpressionType::kScaled: {
        const zk::ScaledExpression<F>* scaled = expression.ToScaled();
        return os << fmt.DebugTuple("Scaled")
                         .Field(*scaled->expr())
                         .Field(scaled->scale())
                         .Finish();
      }
    }
    NOTREACHED();
    return os;
  }
};

template <typename F>
class RustDebugStringifier<std::vector<std::unique_ptr<zk::Expression<F>>>> {
 public:
  static std::ostream& AppendToStream(
      std::ostream& os, RustFormatter& fmt,
      const std::vector<std::unique_ptr<zk::Expression<F>>>& expressions) {
    base::internal::DebugList list = fmt.DebugList();
    for (const std::unique_ptr<zk::Expression<F>>& expression : expressions) {
      list.Entry(*expression);
    }
    return os << list.Finish();
  }
};

}  // namespace tachyon::base::internal

#endif  // TACHYON_ZK_EXPRESSIONS_EXPRESSION_STRINGIFIER_H_
