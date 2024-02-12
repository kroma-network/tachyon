#include "tachyon/zk/expressions/instance_expression.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::zk {

using Fr = math::bn254::Fr;

class InstanceExpressionTest : public math::FiniteFieldTest<Fr> {};

TEST_F(InstanceExpressionTest, DegreeComplexity) {
  std::unique_ptr<InstanceExpression<Fr>> expr =
      InstanceExpression<Fr>::CreateForTesting(
          plonk::InstanceQuery(1, Rotation(1), plonk::InstanceColumnKey(1)));
  EXPECT_EQ(expr->Degree(), size_t{1});
  EXPECT_EQ(expr->Complexity(), uint64_t{1});
}

}  // namespace tachyon::zk
