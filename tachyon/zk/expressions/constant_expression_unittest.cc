#include "tachyon/zk/expressions/constant_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::zk {

using Fr = math::bn254::Fr;

class ConstantExpressionTest : public math::FiniteFieldTest<Fr> {};

TEST_F(ConstantExpressionTest, DegreeComplexity) {
  std::unique_ptr<ConstantExpression<Fr>> expr =
      ConstantExpression<Fr>::CreateForTesting(Fr::One());
  EXPECT_EQ(expr->Degree(), size_t{0});
  EXPECT_EQ(expr->Complexity(), uint64_t{0});
}

}  // namespace tachyon::zk
