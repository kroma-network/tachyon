#include "tachyon/zk/plonk/circuit/expressions/fixed_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::zk {

using Fr = math::bn254::Fr;

TEST(FixedExpressionTest, DegreeComplexity) {
  std::unique_ptr<FixedExpression<Fr>> expr =
      FixedExpression<Fr>::CreateForTesting(FixedQuery(1, 1, Rotation(1)));
  EXPECT_EQ(expr->Degree(), size_t{1});
  EXPECT_EQ(expr->Complexity(), uint64_t{1});
}

}  // namespace tachyon::zk
