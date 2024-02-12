#include "tachyon/zk/expressions/product_expression.h"

#include <algorithm>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/zk/expressions/constant_expression.h"
#include "tachyon/zk/expressions/selector_expression.h"

namespace tachyon::zk {

using Fr = math::bn254::Fr;

class ProductExpressionTest : public math::FiniteFieldTest<Fr> {};

TEST_F(ProductExpressionTest, DegreeComplexity) {
  std::unique_ptr<ConstantExpression<Fr>> left =
      ConstantExpression<Fr>::CreateForTesting(Fr::One());
  std::unique_ptr<SelectorExpression<Fr>> right =
      SelectorExpression<Fr>::CreateForTesting(plonk::Selector::Simple(1));

  size_t left_degree = left->Degree();
  uint64_t left_complexity = left->Complexity();
  size_t right_degree = right->Degree();
  uint64_t right_complexity = right->Complexity();

  std::unique_ptr<ProductExpression<Fr>> prod_expression =
      ProductExpression<Fr>::CreateForTesting(std::move(left),
                                              std::move(right));

  EXPECT_EQ(prod_expression->Degree(), std::max(left_degree, right_degree));
  EXPECT_EQ(
      prod_expression->Complexity(),
      left_complexity + right_complexity + ProductExpression<Fr>::kOverhead);
}

}  // namespace tachyon::zk
