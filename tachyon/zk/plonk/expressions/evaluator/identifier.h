#ifndef TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_IDENTIFIER_H_
#define TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_IDENTIFIER_H_

#include <sstream>
#include <string>

#include "tachyon/base/logging.h"
#include "tachyon/zk/expressions/constant_expression.h"
#include "tachyon/zk/expressions/expression.h"
#include "tachyon/zk/expressions/negated_expression.h"
#include "tachyon/zk/expressions/product_expression.h"
#include "tachyon/zk/expressions/scaled_expression.h"
#include "tachyon/zk/expressions/sum_expression.h"
#include "tachyon/zk/plonk/expressions/advice_expression.h"
#include "tachyon/zk/plonk/expressions/challenge_expression.h"
#include "tachyon/zk/plonk/expressions/fixed_expression.h"
#include "tachyon/zk/plonk/expressions/instance_expression.h"
#include "tachyon/zk/plonk/expressions/selector_expression.h"

namespace tachyon::zk::plonk {

template <typename F>
std::string GetIdentifier(const Expression<F>* input) {
  std::ostringstream ss;
  switch (input->type()) {
    case ExpressionType::kConstant:
      ss << input->ToConstant()->value().ToString();
      return ss.str();
    case ExpressionType::kNegated:
      ss << "(-";
      ss << GetIdentifier(input->ToNegated()->expr());
      ss << ")";
      return ss.str();
    case ExpressionType::kSum:
      ss << "(";
      ss << GetIdentifier(input->ToSum()->left());
      ss << "+";
      ss << GetIdentifier(input->ToSum()->right());
      ss << ")";
      return ss.str();
    case ExpressionType::kProduct:
      ss << "(";
      ss << GetIdentifier(input->ToProduct()->left());
      ss << "*";
      ss << GetIdentifier(input->ToProduct()->right());
      ss << ")";
      return ss.str();
    case ExpressionType::kScaled:
      ss << "*" << input->ToScaled()->scale().ToString();
      return ss.str();
    case ExpressionType::kFixed: {
      const FixedExpression<F>* fixed = input->ToFixed();
      ss << "fixed[" << fixed->query().column().index() << "]["
         << fixed->query().rotation().value() << "]";
      return ss.str();
    }
    case ExpressionType::kAdvice: {
      const AdviceExpression<F>* advice = input->ToAdvice();
      ss << "advice[" << advice->query().column().index() << "]["
         << advice->query().rotation().value() << "]";
      return ss.str();
    }
    case ExpressionType::kInstance: {
      const InstanceExpression<F>* instance = input->ToInstance();
      ss << "instance[" << instance->query().column().index() << "]["
         << instance->query().rotation().value() << "]";
      return ss.str();
    }
    case ExpressionType::kChallenge: {
      const ChallengeExpression<F>* challenge = input->ToChallenge();
      ss << "challenge[" << challenge->challenge().index() << "]";
      return ss.str();
    }
    case ExpressionType::kSelector: {
      const SelectorExpression<F>* selector = input->ToSelector();
      ss << "selector[" << selector->selector().index() << "]";
      return ss.str();
    }
    case ExpressionType::kFirstRow:
    case ExpressionType::kLastRow:
    case ExpressionType::kTransition:
    case ExpressionType::kVariable:
      NOTREACHED() << "AIR expression " << ExpressionTypeToString(input->type())
                   << " is not allowed in plonk!";
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_IDENTIFIER_H_
