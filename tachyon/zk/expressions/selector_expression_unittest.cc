#include "tachyon/zk/expressions/selector_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk {

using F = math::GF7;

class SelectorExpressionTest : public math::FiniteFieldTest<F> {};

TEST_F(SelectorExpressionTest, Degree_Complexity) {
  std::unique_ptr<SelectorExpression<F>> expr =
      SelectorExpression<F>::CreateForTesting(plonk::Selector::Simple(1));
  EXPECT_EQ(expr->Degree(), size_t{1});
  EXPECT_EQ(expr->Complexity(), uint64_t{1});
}

}  // namespace tachyon::zk
