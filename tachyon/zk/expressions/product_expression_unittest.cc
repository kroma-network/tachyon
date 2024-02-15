#include "tachyon/zk/expressions/product_expression.h"

#include <algorithm>

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/zk/expressions/constant_expression.h"
#include "tachyon/zk/expressions/selector_expression.h"

namespace tachyon::zk {

using F = math::GF7;

class ProductExpressionTest : public math::FiniteFieldTest<F> {};

TEST_F(ProductExpressionTest, DegreeComplexity) {
  std::unique_ptr<ConstantExpression<F>> left =
      ConstantExpression<F>::CreateForTesting(F::One());
  std::unique_ptr<SelectorExpression<F>> right =
      SelectorExpression<F>::CreateForTesting(plonk::Selector::Simple(1));

  size_t left_degree = left->Degree();
  uint64_t left_complexity = left->Complexity();
  size_t right_degree = right->Degree();
  uint64_t right_complexity = right->Complexity();

  std::unique_ptr<ProductExpression<F>> prod_expression =
      ProductExpression<F>::CreateForTesting(std::move(left), std::move(right));

  EXPECT_EQ(prod_expression->Degree(), std::max(left_degree, right_degree));
  EXPECT_EQ(prod_expression->Complexity(), left_complexity + right_complexity +
                                               ProductExpression<F>::kOverhead);
}

}  // namespace tachyon::zk
