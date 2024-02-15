#include "tachyon/zk/expressions/negated_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/zk/expressions/constant_expression.h"

namespace tachyon::zk {

using F = math::GF7;

class NegatedExpressionTest : public math::FiniteFieldTest<F> {};

TEST_F(NegatedExpressionTest, DegreeComplexity) {
  std::unique_ptr<ConstantExpression<F>> expr =
      ConstantExpression<F>::CreateForTesting(F::One());

  size_t expr_degree = expr->Degree();
  uint64_t expr_complexity = expr->Complexity();

  std::unique_ptr<NegatedExpression<F>> negated_expression =
      NegatedExpression<F>::CreateForTesting(std::move(expr));

  EXPECT_EQ(negated_expression->Degree(), expr_degree);
  EXPECT_EQ(negated_expression->Complexity(),
            expr_complexity + NegatedExpression<F>::kOverhead);
}

}  // namespace tachyon::zk
