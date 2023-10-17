#include "tachyon/zk/plonk/circuit/expressions/challenge_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::zk {

using Fr = math::bn254::Fr;

TEST(ChallengeExpressionTest, DegreeComplexity) {
  std::unique_ptr<ChallengeExpression<Fr>> expr =
      ChallengeExpression<Fr>::CreateForTesting(Challenge(1, Phase(1)));
  EXPECT_EQ(expr->Degree(), size_t{0});
  EXPECT_EQ(expr->Complexity(), uint64_t{0});
}

}  // namespace tachyon::zk
