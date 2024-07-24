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
class Identifier : public Evaluator<F, void> {
 public:
  explicit Identifier(std::ostringstream& ss) : ss_(ss) {}

  // Evaluator methods
  void Evaluate(const Expression<F>* input) override {
    switch (input->type()) {
      case ExpressionType::kConstant:
        ss_ << input->ToConstant()->value().ToString();
        return;
      case ExpressionType::kNegated:
        ss_ << "(-";
        Evaluate(input->ToNegated()->expr());
        ss_ << ")";
        return;
      case ExpressionType::kSum:
        ss_ << "(";
        Evaluate(input->ToSum()->left());
        ss_ << "+";
        Evaluate(input->ToSum()->right());
        ss_ << ")";
        return;
      case ExpressionType::kProduct:
        ss_ << "(";
        Evaluate(input->ToProduct()->left());
        ss_ << "*";
        Evaluate(input->ToProduct()->right());
        ss_ << ")";
        return;
      case ExpressionType::kScaled:
        ss_ << "*";
        ss_ << input->ToScaled()->scale().ToString();
        return;
      case ExpressionType::kFixed: {
        const FixedExpression<F>* fixed = input->ToFixed();
        ss_ << "fixed[" << fixed->query().column().index() << "]["
            << fixed->query().rotation().value() << "]";
        return;
      }
      case ExpressionType::kAdvice: {
        const AdviceExpression<F>* advice = input->ToAdvice();
        ss_ << "advice[" << advice->query().column().index() << "]["
            << advice->query().rotation().value() << "]";
        return;
      }
      case ExpressionType::kInstance: {
        const InstanceExpression<F>* instance = input->ToInstance();
        ss_ << "instance[" << instance->query().column().index() << "]["
            << instance->query().rotation().value() << "]";
        return;
      }
      case ExpressionType::kChallenge: {
        const ChallengeExpression<F>* challenge = input->ToChallenge();
        ss_ << "challenge[" << challenge->challenge().index() << "]";
        return;
      }
      case ExpressionType::kSelector: {
        const SelectorExpression<F>* selector = input->ToSelector();
        ss_ << "selector[" << selector->selector().index() << "]";
        return;
      }
      case ExpressionType::kFirstRow:
      case ExpressionType::kLastRow:
      case ExpressionType::kTransition:
      case ExpressionType::kVariable:
        NOTREACHED() << "AIR expression "
                     << ExpressionTypeToString(input->type())
                     << " is not allowed in plonk!";
    }
    NOTREACHED();
  }

 private:
  std::ostringstream& ss_;
};

template <typename F>
std::string GetIdentifier(const Expression<F>* input) {
  std::ostringstream ss;
  Identifier<F> identifier(ss);
  identifier.Evaluate(input);
  return ss.str();
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_EXPRESSIONS_EVALUATOR_IDENTIFIER_H_
