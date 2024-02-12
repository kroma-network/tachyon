#include "tachyon/zk/expressions/scaled_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/zk/expressions/constant_expression.h"

namespace tachyon::zk {

using F = math::GF7;

class ScaledExpressionTest : public math::FiniteFieldTest<F> {};

TEST_F(ScaledExpressionTest, DegreeComplexity) {
  F scale = F::One();
  std::unique_ptr<ConstantExpression<F>> expr =
      ConstantExpression<F>::CreateForTesting(F::One());

  size_t expr_degree = expr->Degree();
  uint64_t expr_complexity = expr->Complexity();

  std::unique_ptr<ScaledExpression<F>> scaled_expr =
      ScaledExpression<F>::CreateForTesting(std::move(expr), std::move(scale));

  EXPECT_EQ(scaled_expr->Degree(), expr_degree);
  EXPECT_EQ(scaled_expr->Complexity(),
            expr_complexity + ScaledExpression<F>::kOverhead);
}

}  // namespace tachyon::zk
