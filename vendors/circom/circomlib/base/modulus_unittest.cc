#include "circomlib/base/modulus.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::circom {

using F = math::bn254::Fr;

class ModulusTest : public math::FiniteFieldTest<F> {};

TEST_F(ModulusTest, BigIntConversions) {
  math::BigInt<4> expected = math::BigInt<4>::Random();
  Modulus field = Modulus::FromBigInt(expected);
  EXPECT_EQ(field.ToBigInt<4>(), expected);
}

TEST_F(ModulusTest, Read) {
  std::array<uint8_t, 8> data;

  {
    base::Uint8VectorBuffer buffer;
    Modulus field;
    // Should return false when it fails to read the field size.
    ASSERT_FALSE(field.Read(buffer));
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{3}));
    buffer.set_buffer_offset(0);
    Modulus field;
    // Should return false when the field size is not a multiple of 8.
    ASSERT_FALSE(field.Read(buffer));
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{8}));
    buffer.set_buffer_offset(0);
    Modulus field;
    // Should return false when it fails to read the field data.
    ASSERT_FALSE(field.Read(buffer));
  }

  {
    base::Uint8VectorBuffer buffer;
    ASSERT_TRUE(buffer.Write(uint32_t{8}));
    ASSERT_TRUE(buffer.Write(data));
    buffer.set_buffer_offset(0);
    Modulus field;
    ASSERT_TRUE(field.Read(buffer));
  }
}

}  // namespace tachyon::circom
