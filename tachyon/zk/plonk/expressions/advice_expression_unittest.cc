#include "tachyon/zk/plonk/expressions/advice_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk::plonk {

using F = math::GF7;

class AdviceExpressionTest : public math::FiniteFieldTest<F> {};

TEST_F(AdviceExpressionTest, DegreeComplexity) {
  std::unique_ptr<AdviceExpression<F>> expr =
      AdviceExpression<F>::CreateForTesting(
          AdviceQuery(1, Rotation(1), AdviceColumnKey(1, plonk::Phase(0))));
  EXPECT_EQ(expr->Degree(), size_t{1});
  EXPECT_EQ(expr->Complexity(), uint64_t{1});
}

}  // namespace tachyon::zk::plonk
