#include "tachyon/zk/expressions/sum_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/zk/expressions/constant_expression.h"

namespace tachyon::zk {

using F = math::GF7;

class SumExpressionTest : public math::FiniteFieldTest<F> {};

TEST_F(SumExpressionTest, DegreeComplexity) {
  std::unique_ptr<ConstantExpression<F>> left =
      ConstantExpression<F>::CreateForTesting(F::One());
  std::unique_ptr<ConstantExpression<F>> right =
      ConstantExpression<F>::CreateForTesting(F::One());

  size_t left_degree = left->Degree();
  uint64_t left_complexity = left->Complexity();
  size_t right_degree = right->Degree();
  uint64_t right_complexity = right->Complexity();

  std::unique_ptr<SumExpression<F>> sum_expression =
      SumExpression<F>::CreateForTesting(std::move(left), std::move(right));

  EXPECT_EQ(sum_expression->Degree(), std::max(left_degree, right_degree));
  EXPECT_EQ(sum_expression->Complexity(),
            left_complexity + right_complexity + SumExpression<F>::kOverhead);
}

}  // namespace tachyon::zk
