#include "tachyon/crypto/sumcheck/multilinear/sumcheck_verifier_msg.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::crypto {
namespace {

using F = math::GF7;

class SumcheckVerifierMsgTest : public math::FiniteFieldTest<math::GF7> {};

}  // namespace

TEST_F(SumcheckVerifierMsgTest, Copyable) {
  SumcheckVerifierMsg<F> expected = {F::Random()};

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  SumcheckVerifierMsg<F> value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::crypto
