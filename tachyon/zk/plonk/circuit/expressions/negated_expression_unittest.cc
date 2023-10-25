#include "tachyon/zk/plonk/circuit/expressions/negated_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/zk/plonk/circuit/expressions/constant_expression.h"

namespace tachyon::zk {

using Fr = math::bn254::Fr;

class NegatedExpressionTest : public testing::Test {
 public:
  static void SetUpTestSuite() { Fr::Init(); }
};

TEST_F(NegatedExpressionTest, DegreeComplexity) {
  std::unique_ptr<ConstantExpression<Fr>> expr =
      ConstantExpression<Fr>::CreateForTesting(Fr::One());

  size_t expr_degree = expr->Degree();
  uint64_t expr_complexity = expr->Complexity();

  std::unique_ptr<NegatedExpression<Fr>> negated_expression =
      NegatedExpression<Fr>::CreateForTesting(std::move(expr));

  EXPECT_EQ(negated_expression->Degree(), expr_degree);
  EXPECT_EQ(negated_expression->Complexity(),
            expr_complexity + NegatedExpression<Fr>::kOverhead);
}

}  // namespace tachyon::zk
