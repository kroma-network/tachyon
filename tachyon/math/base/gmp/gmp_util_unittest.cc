#include "tachyon/math/base/gmp/gmp_util.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/logging.h"
#include "tachyon/build/build_config.h"

namespace tachyon {
namespace math {
namespace gmp {

TEST(GmpUtilTest, FromDecString) {
  EXPECT_EQ(FromDecString("1234"), mpz_class(1234));
  EXPECT_EQ(FromDecString("-1234"), mpz_class(-1234));
}

TEST(GmpUtilTest, FromHexString) {
  EXPECT_EQ(FromHexString("0x1234"), mpz_class(0x1234));
  EXPECT_EQ(FromHexString("1234"), mpz_class(0x1234));
}

TEST(GmpUtilTest, FromUnsignedInt) {
  EXPECT_EQ(FromUnsignedInt(static_cast<unsigned int>(1234)), mpz_class(1234));
}

TEST(GmpUtilTest, FromSignedInt) {
  EXPECT_EQ(FromSignedInt(1234), mpz_class(1234));
  EXPECT_EQ(FromSignedInt(-1234), mpz_class(-1234));
}

TEST(GmpUtilTest, Sign) {
  mpz_class zero(0);
  EXPECT_EQ(GetSign(zero), Sign::kZero);
  EXPECT_TRUE(IsZero(zero));
  EXPECT_FALSE(IsNegative(zero));
  EXPECT_FALSE(IsPositive(zero));

  mpz_class positive(1234);
  EXPECT_EQ(GetSign(positive), Sign::kPositive);
  EXPECT_FALSE(IsZero(positive));
  EXPECT_FALSE(IsNegative(positive));
  EXPECT_TRUE(IsPositive(positive));

  mpz_class negative(-1234);
  EXPECT_EQ(GetSign(negative), Sign::kNegative);
  EXPECT_FALSE(IsZero(negative));
  EXPECT_TRUE(IsNegative(negative));
  EXPECT_FALSE(IsPositive(negative));
}

TEST(GmpUtilTest, Abs) {
  mpz_class zero(0);
  EXPECT_EQ(GetAbs(zero), zero);

  mpz_class positive(1234);
  EXPECT_EQ(GetAbs(positive), positive);

  mpz_class negative(-1234);
  EXPECT_EQ(GetAbs(negative), positive);
}

TEST(GmpUtilTest, Bits) {
  mpz_class value(0b11001);
  EXPECT_EQ(GetNumBits(value), 64);

#if ARCH_CPU_LITTLE_ENDIAN
  EXPECT_EQ(TestBit(value, 0), true);
  EXPECT_EQ(TestBit(value, 1), false);
  EXPECT_EQ(TestBit(value, 2), false);
  EXPECT_EQ(TestBit(value, 3), true);
  EXPECT_EQ(TestBit(value, 4), true);
#else
  EXPECT_EQ(TestBit(value, 0), true);
  EXPECT_EQ(TestBit(value, 1), true);
  EXPECT_EQ(TestBit(value, 2), false);
  EXPECT_EQ(TestBit(value, 3), false);
  EXPECT_EQ(TestBit(value, 4), true);
#endif
}

TEST(GmpUtilTest, Limbs) {
  mpz_class value(1234);
  EXPECT_EQ(GetLimbSize(value), static_cast<size_t>(1));
  EXPECT_EQ(GetLimb(value, 0), 1234);
}

}  // namespace gmp
}  // namespace math
}  // namespace tachyon
