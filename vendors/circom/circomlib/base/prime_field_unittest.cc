#include "circomlib/base/prime_field.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::circom {

using F = math::bn254::Fr;

class PrimeFieldTest : public math::FiniteFieldTest<F> {};

TEST_F(PrimeFieldTest, BigIntConversions) {
  math::BigInt<4> expected = math::BigInt<4>::Random();
  PrimeField field = PrimeField::FromBigInt(expected);
  EXPECT_EQ(field.ToBigInt<4>(), expected);
}

TEST_F(PrimeFieldTest, PrimeFieldConversions) {
  F expected = F::Random();
  {
    PrimeField field = PrimeField::FromNative<true>(expected);
    F actual = field.ToNative<true, F>();
    EXPECT_EQ(actual, expected);
  }
  {
    PrimeField field = PrimeField::FromNative<false>(expected);
    F actual = field.ToNative<false, F>();
    EXPECT_EQ(actual, expected);
  }
}

TEST_F(PrimeFieldTest, Read) {
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
