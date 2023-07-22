#include "tachyon/math/finite_fields/modulus.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"

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

TEST(ModulusTest, HasSpareBit) {
  BigInt<1> modulus = BigInt<1>::FromHexString("0x8000000000000000");
  EXPECT_FALSE(Modulus<1>::HasSpareBit(modulus));
  modulus = BigInt<1>::FromHexString("0x7000000000000000");
  EXPECT_TRUE(Modulus<1>::HasSpareBit(modulus));
}

TEST(ModulusTest, Montgomery) {
  EXPECT_EQ(Modulus<4>::MontgomeryR(bn254::FqConfig::kModulus),
            BigInt<4>({
                UINT64_C(15230403791020821917),
                UINT64_C(754611498739239741),
                UINT64_C(7381016538464732716),
                UINT64_C(1011752739694698287),
            }));
  EXPECT_EQ(Modulus<4>::MontgomeryR2(bn254::FqConfig::kModulus),
            BigInt<4>({
                UINT64_C(17522657719365597833),
                UINT64_C(13107472804851548667),
                UINT64_C(5164255478447964150),
                UINT64_C(493319470278259999),
            }));
}

TEST(ModulusTest, Inverse) {
  EXPECT_EQ(Modulus<4>::Inverse<uint32_t>(bn254::FqConfig::kModulus),
            UINT32_C(3834012553));
  EXPECT_EQ(Modulus<4>::Inverse<uint64_t>(bn254::FqConfig::kModulus),
            UINT64_C(9786893198990664585));
}

}  // namespace math
}  // namespace tachyon
