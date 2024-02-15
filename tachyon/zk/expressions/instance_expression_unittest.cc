#include "tachyon/zk/expressions/instance_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk {

using F = math::GF7;

class InstanceExpressionTest : public math::FiniteFieldTest<F> {};

TEST_F(InstanceExpressionTest, DegreeComplexity) {
  std::unique_ptr<InstanceExpression<F>> expr =
      InstanceExpression<F>::CreateForTesting(
          plonk::InstanceQuery(1, Rotation(1), plonk::InstanceColumnKey(1)));
  EXPECT_EQ(expr->Degree(), size_t{1});
  EXPECT_EQ(expr->Complexity(), uint64_t{1});
}

}  // namespace tachyon::zk
