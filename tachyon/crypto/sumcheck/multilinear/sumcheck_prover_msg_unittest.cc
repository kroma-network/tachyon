#include "tachyon/crypto/sumcheck/multilinear/sumcheck_prover_msg.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"

namespace tachyon::crypto {
namespace {

using F = math::GF7;

class SumcheckProverMsgTest : public math::FiniteFieldTest<math::GF7> {};

}  // namespace

TEST_F(SumcheckProverMsgTest, Copyable) {
  SumcheckProverMsg<F, 5> expected = {
      math::UnivariateEvaluations<F, 5>::Random(5)};

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  SumcheckProverMsg<F, 5> value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::crypto
