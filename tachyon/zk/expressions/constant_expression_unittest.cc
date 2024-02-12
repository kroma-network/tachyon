#include "tachyon/zk/expressions/constant_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk {

using F = math::GF7;

class ConstantExpressionTest : public math::FiniteFieldTest<F> {};

TEST_F(ConstantExpressionTest, DegreeComplexity) {
  std::unique_ptr<ConstantExpression<F>> expr =
      ConstantExpression<F>::CreateForTesting(F::One());
  EXPECT_EQ(expr->Degree(), size_t{0});
  EXPECT_EQ(expr->Complexity(), uint64_t{0});
}

}  // namespace tachyon::zk
