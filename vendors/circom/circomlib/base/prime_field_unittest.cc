#include "circomlib/base/prime_field.h"

#include "gtest/gtest.h"

namespace tachyon::circom {

TEST(PrimeFieldTest, Conversions) {
  math::BigInt<4> expected = math::BigInt<4>::Random();
  PrimeField field = PrimeField::FromBigInt(expected);
  EXPECT_EQ(field.ToBigInt<4>(), expected);
}

}  // namespace tachyon::circom
