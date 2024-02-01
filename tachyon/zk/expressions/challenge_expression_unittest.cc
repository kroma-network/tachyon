#include "tachyon/zk/expressions/challenge_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::zk {

using Fr = math::bn254::Fr;

class ChallengeExpressionTest : public testing::Test {
 public:
  static void SetUpTestSuite() { Fr::Init(); }
};

TEST_F(ChallengeExpressionTest, DegreeComplexity) {
  std::unique_ptr<ChallengeExpression<Fr>> expr =
      ChallengeExpression<Fr>::CreateForTesting(
          plonk::Challenge(1, plonk::Phase(1)));
  EXPECT_EQ(expr->Degree(), size_t{0});
  EXPECT_EQ(expr->Complexity(), uint64_t{0});
}

}  // namespace tachyon::zk
