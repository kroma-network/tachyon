#include "tachyon/zk/plonk/expressions/challenge_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk::plonk {

using F = math::GF7;

class ChallengeExpressionTest : public math::FiniteFieldTest<F> {};

TEST_F(ChallengeExpressionTest, DegreeComplexity) {
  std::unique_ptr<ChallengeExpression<F>> expr =
      ChallengeExpression<F>::CreateForTesting(Challenge(1, Phase(1)));
  EXPECT_EQ(expr->Degree(), size_t{0});
  EXPECT_EQ(expr->Complexity(), uint64_t{0});
}

}  // namespace tachyon::zk::plonk
