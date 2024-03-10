#include "circomlib/base/prime_field.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/buffer/vector_buffer.h"

namespace tachyon::circom {

TEST(PrimeFieldTest, Conversions) {
  math::BigInt<4> expected = math::BigInt<4>::Random();
  PrimeField field = PrimeField::FromBigInt(expected);
  EXPECT_EQ(field.ToBigInt<4>(), expected);
}

TEST(PrimeFieldTest, Read) {
  std::array<uint8_t, 8> data;

  {
    base::Uint8VectorBuffer buffer;
    PrimeField field;
    // Should return false when it fails to read the field size.
    ASSERT_FALSE(field.Read(buffer));
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{3}));
    buffer.set_buffer_offset(0);
    PrimeField field;
    // Should return false when the field size is not a multiple of 8.
    ASSERT_FALSE(field.Read(buffer));
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{8}));
    buffer.set_buffer_offset(0);
    PrimeField field;
    // Should return false when it fails to read the field data.
    ASSERT_FALSE(field.Read(buffer));
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{8}));
    ASSERT_TRUE(buffer.Write(data));
    buffer.set_buffer_offset(0);
    PrimeField field;
    ASSERT_TRUE(field.Read(buffer));
  }

  {
    base::Uint8VectorBuffer buffer;
    PrimeField field;
    // Should return false when it fails to read the field data.
    ASSERT_FALSE(field.Read(buffer, 8));
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(data));
    buffer.set_buffer_offset(0);
    PrimeField field;
    ASSERT_TRUE(field.Read(buffer, 8));
  }
}

}  // namespace tachyon::circom
