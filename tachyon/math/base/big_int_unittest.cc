#include "tachyon/math/base/big_int.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

TEST(BigIntTest, DecString) {
  // 1 << 65
  BigInt<2> big_int = BigInt<2>::FromDecString("36893488147419103232");
  EXPECT_EQ(big_int.ToString(), "36893488147419103232");
}

TEST(BigIntTest, HexString) {
  {
    // 1 << 65
    BigInt<2> big_int = BigInt<2>::FromHexString("20000000000000000");
    EXPECT_EQ(big_int.ToHexString(), "0x20000000000000000");
  }
  {
    // 1 << 65
    BigInt<2> big_int = BigInt<2>::FromHexString("0x20000000000000000");
    EXPECT_EQ(big_int.ToHexString(), "0x20000000000000000");
  }
}

TEST(BigIntTest, Comparison) {
  // 1 << 65
  BigInt<2> big_int = BigInt<2>::FromHexString("20000000000000000");
  BigInt<2> big_int2 = BigInt<2>::FromHexString("20000000000000001");
  EXPECT_TRUE(big_int == big_int);
  EXPECT_TRUE(big_int != big_int2);
  EXPECT_TRUE(big_int < big_int2);
  EXPECT_TRUE(big_int <= big_int2);
  EXPECT_TRUE(big_int2 > big_int);
  EXPECT_TRUE(big_int2 >= big_int);
}

}  // namespace math
}  // namespace tachyon
