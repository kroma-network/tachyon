#ifndef TACHYON_ZK_EXPRESSIONS_EVALUATOR_IDENTIFIER_H_
#define TACHYON_ZK_EXPRESSIONS_EVALUATOR_IDENTIFIER_H_

#include <sstream>
#include <string>

#include "tachyon/base/logging.h"
#include "tachyon/zk/expressions/advice_expression.h"
#include "tachyon/zk/expressions/challenge_expression.h"
#include "tachyon/zk/expressions/constant_expression.h"
#include "tachyon/zk/expressions/expression.h"
#include "tachyon/zk/expressions/expression_type.h"
#include "tachyon/zk/expressions/fixed_expression.h"
#include "tachyon/zk/expressions/instance_expression.h"
#include "tachyon/zk/expressions/negated_expression.h"
#include "tachyon/zk/expressions/product_expression.h"
#include "tachyon/zk/expressions/scaled_expression.h"
#include "tachyon/zk/expressions/selector_expression.h"
#include "tachyon/zk/expressions/sum_expression.h"

namespace tachyon::zk {

template <typename F>
std::string Identifier(const Expression<F>* input) {
  std::ostringstream ss;
  switch (input->type()) {
    case ExpressionType::kConstant:
      ss << input->ToConstant()->value().ToString();
      return ss.str();
    case ExpressionType::kFixed:
      ss << "fixed[" << input->ToFixed()->query().column().index() << "]["
         << input->ToFixed()->query().rotation().value() << "]";
      return ss.str();
    case ExpressionType::kAdvice:
      ss << "advice[" << input->ToAdvice()->query().column().index() << "]["
         << input->ToFixed()->query().rotation().value() << "]";
      return ss.str();
    case ExpressionType::kInstance:
      ss << "instance[" << input->ToInstance()->query().column().index() << "]["
         << input->ToFixed()->query().rotation().value() << "]";
      return ss.str();
    case ExpressionType::kChallenge:
      ss << "challenge[" << input->ToChallenge()->challenge().index() << "]";
      return ss.str();
    case ExpressionType::kSelector:
      ss << "selector[" << input->ToSelector()->selector().index() << "]";
      return ss.str();
    case ExpressionType::kNegated:
      ss << "(-";
      ss << Identifier(input->ToNegated()->expr());
      ss << ")";
      return ss.str();
    case ExpressionType::kSum:
      ss << "(";
      ss << Identifier(input->ToSum()->left());
      ss << "+";
      ss << Identifier(input->ToSum()->right());
      ss << ")";
      return ss.str();
    case ExpressionType::kProduct:
      ss << "(";
      ss << Identifier(input->ToProduct()->left());
      ss << "*";
      ss << Identifier(input->ToProduct()->right());
      ss << ")";
      return ss.str();
    case ExpressionType::kScaled:
      ss << "*" << input->ToScaled()->scale().ToString();
      return ss.str();
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_EXPRESSIONS_EVALUATOR_IDENTIFIER_H_
