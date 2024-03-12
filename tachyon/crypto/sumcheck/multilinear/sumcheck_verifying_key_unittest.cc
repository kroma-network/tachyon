#include "tachyon/crypto/sumcheck/multilinear/sumcheck_verifying_key.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"

namespace tachyon::crypto {

TEST(SumcheckVerifyingKeyTest, Copyable) {
  SumcheckVerifyingKey expected = SumcheckVerifyingKey::Random();

  base::Uint8VectorBuffer write_buf;
  ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
  ASSERT_TRUE(write_buf.Write(expected));
  ASSERT_TRUE(write_buf.Done());

  write_buf.set_buffer_offset(0);

  SumcheckVerifyingKey value;
  ASSERT_TRUE(write_buf.Read(&value));
  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::crypto
