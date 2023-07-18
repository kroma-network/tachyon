#include "tachyon/math/finite_fields/modulus.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

TEST(ModulusTest, CanUseNoCarryMulOptimization) {
  BigInt<1> modulus = BigInt<1>::FromHexString("0x8000000000000000");
  EXPECT_FALSE(Modulus<1>::CanUseNoCarryMulOptimization(modulus));
  modulus = BigInt<1>::FromHexString("0x7fffffffffffffff");
  EXPECT_FALSE(Modulus<1>::CanUseNoCarryMulOptimization(modulus));
  modulus = BigInt<1>::FromHexString("0x7000000000000000");
  EXPECT_TRUE(Modulus<1>::CanUseNoCarryMulOptimization(modulus));
}

TEST(ModulusTest, HasSparseBit) {
  BigInt<1> modulus = BigInt<1>::FromHexString("0x8000000000000000");
  EXPECT_FALSE(Modulus<1>::HasSparseBit(modulus));
  modulus = BigInt<1>::FromHexString("0x7000000000000000");
  EXPECT_TRUE(Modulus<1>::HasSparseBit(modulus));
}

TEST(ModulusTest, Montgomery) {
  BigInt<1> modulus(59);
  EXPECT_EQ(Modulus<1>::MontgomeryR(modulus), BigInt<1>(5));
  EXPECT_EQ(Modulus<1>::MontgomeryR2(modulus), BigInt<1>(25));
}

TEST(ModulusTest, Inverse) {
  BigInt<1> modulus(59);
  EXPECT_EQ(Modulus<1>::Inverse(modulus), UINT64_C(3751880150584993549));
}

}  // namespace math
}  // namespace tachyon
