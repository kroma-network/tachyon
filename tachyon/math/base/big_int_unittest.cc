#include "tachyon/math/base/big_int.h"

#include <vector>

#include "gtest/gtest.h"

namespace tachyon::math {

TEST(BigIntTest, Zero) {
  BigInt<2> big_int = BigInt<2>::Zero();
  EXPECT_TRUE(big_int.IsZero());
  EXPECT_FALSE(big_int.IsOne());
  EXPECT_TRUE(big_int.IsEven());
  EXPECT_FALSE(big_int.IsOdd());
}

TEST(BigIntTest, One) {
  BigInt<2> big_int = BigInt<2>::One();
  EXPECT_FALSE(big_int.IsZero());
  EXPECT_TRUE(big_int.IsOne());
  EXPECT_FALSE(big_int.IsEven());
  EXPECT_TRUE(big_int.IsOdd());
}

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

TEST(BigIntTest, BitsLEConversion) {
  std::bitset<255> input(
      "011101111110011110110101010100110010011011110111011101000111010111110011"
      "000100011000011100111011011100111101100101100111001101011010000011111110"
      "000010011110011110001011111101111001100001100000111010000101111101010010"
      "101011110101110101011101011001100110000");
  BigInt<4> big_int = BigInt<4>::FromBitsLE(input);
  ASSERT_EQ(big_int,
            BigInt<4>::FromDecString("27117311055620256798560880810000042840428"
                                     "971800021819916023577129547249660720"));
  EXPECT_EQ(big_int.ToBitsLE<255>(), input);
}

TEST(BigIntTest, BitsBEConversion) {
  std::bitset<255> input(
      "000011001100110101110101011101011110101010010101111101000010111000001100"
      "001100111101111110100011110011110010000011111110000010110101100111001101"
      "001101111001110110111001110000110001000110011111010111000101110111011110"
      "1100100110010101010110111100111111011100110000");
  BigInt<4> big_int = BigInt<4>::FromBitsBE(input);
  ASSERT_EQ(big_int,
            BigInt<4>::FromDecString("27117311055620256798560880810000042840428"
                                     "971800021819916023577129547249660720"));
  EXPECT_EQ(big_int.ToBitsBE<255>(), input);
}

TEST(BigIntTest, BytesLEConversion) {
  std::vector<uint8_t> input{48,  179, 174, 174, 87,  169, 47,  116,
                             48,  204, 251, 197, 243, 4,   127, 208,
                             154, 179, 236, 185, 157, 195, 136, 249,
                             58,  186, 123, 147, 169, 218, 243, 59};

  BigInt<4> big_int = BigInt<4>::FromBytesLE(input);
  ASSERT_EQ(big_int,
            BigInt<4>::FromDecString("27117311055620256798560880810000042840428"
                                     "971800021819916023577129547249660720"));
  EXPECT_EQ(big_int.ToBytesLE(), input);
}

TEST(BigIntTest, BytesBEConversion) {
  std::vector<uint8_t> input{59,  243, 218, 169, 147, 123, 186, 58,
                             249, 136, 195, 157, 185, 236, 179, 154,
                             208, 127, 4,   243, 197, 251, 204, 48,
                             116, 47,  169, 87,  174, 174, 179, 48};
  BigInt<4> big_int = BigInt<4>::FromBytesBE(input);
  ASSERT_EQ(big_int,
            BigInt<4>::FromDecString("27117311055620256798560880810000042840428"
                                     "971800021819916023577129547249660720"));
  EXPECT_EQ(big_int.ToBytesBE(), input);
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

TEST(BigIntTest, ExtractBits) {
  BigInt<2> big_int = BigInt<2>::Random();
  size_t bit_count = 4;
  for (size_t offset = 0; offset < 16; offset += bit_count) {
    uint64_t value = 0;
    for (size_t i = 0; i < bit_count; ++i) {
      if (BitTraits<BigInt<2>>::TestBit(big_int, offset + i)) {
        value |= UINT64_C(1) << i;
      }
    }
    EXPECT_EQ(big_int.ExtractBits64(offset, bit_count), value);
    EXPECT_EQ(big_int.ExtractBits32(offset, bit_count), value);
  }
}

TEST(BigIntTest, Operations) {
  BigInt<2> big_int =
      BigInt<2>::FromDecString("123456789012345678909876543211235312");
  BigInt<2> big_int2 =
      BigInt<2>::FromDecString("734581237591230158128731489729873983");
  {
    uint64_t carry = 0;
    BigInt<2> a = big_int;
    BigInt<2> sum =
        BigInt<2>::FromDecString("858038026603575837038608032941109295");
    BigInt<2> b = big_int2;
    EXPECT_EQ(a.AddInPlace(big_int2, carry), sum);
    EXPECT_EQ(carry, 0);
    EXPECT_EQ(b.AddInPlace(big_int, carry), sum);
    EXPECT_EQ(carry, 0);
  }
  {
    uint64_t borrow = 0;
    BigInt<2> a = big_int;
    BigInt<2> amb =
        BigInt<2>::FromDecString("339671242472359578984155752485249572785");
    EXPECT_EQ(a.SubInPlace(big_int2, borrow), amb);
    EXPECT_EQ(borrow, 1);
    BigInt<2> b = big_int2;
    BigInt<2> bma =
        BigInt<2>::FromDecString("611124448578884479218854946518638671");
    EXPECT_EQ(b.SubInPlace(big_int, borrow), bma);
    EXPECT_EQ(borrow, 0);
  }
  {
    uint64_t carry = 0;
    BigInt<2> a = big_int;
    BigInt<2> mulby2 =
        BigInt<2>::FromDecString("246913578024691357819753086422470624");
    EXPECT_EQ(a.MulBy2InPlace(carry), mulby2);
    EXPECT_EQ(carry, 0);
  }
  {
    uint64_t carry = 0;
    BigInt<2> a = big_int;
    BigInt<2> mulbyn =
        BigInt<2>::FromDecString("3950617248395061725116049382759529984");
    EXPECT_EQ(a.MulBy2ExpInPlace(5), mulbyn);
    EXPECT_EQ(carry, 0);
  }
  {
    BigInt<2> a = big_int;
    BigInt<2> b = big_int2;
    BigInt<2> hi;
    a.MulInPlace(b, hi);
    EXPECT_EQ(
        a, BigInt<2>::FromDecString("335394729415762779748307316131549975568"));
    EXPECT_EQ(hi,
              BigInt<2>::FromDecString("266511138036132956757991041665338"));
  }
  {
    BigInt<2> a = big_int;
    BigInt<2> divby2 =
        BigInt<2>::FromDecString("61728394506172839454938271605617656");
    EXPECT_EQ(a.DivBy2InPlace(), divby2);
  }
  {
    BigInt<2> a = big_int;
    BigInt<2> divbyn =
        BigInt<2>::FromDecString("3858024656635802465933641975351103");
    EXPECT_EQ(a.DivBy2ExpInPlace(5), divbyn);
  }
  {
    BigInt<2> a = big_int;
    BigInt<2> b = big_int2;
    DivResult<BigInt<2>> adb = {
        BigInt<2>(),
        a,
    };
    EXPECT_EQ(a.Divide(b), adb);
    DivResult<BigInt<2>> bda = {
        BigInt<2>(5),
        BigInt<2>::FromDecString("117297292529501763579348773673697423"),
    };
    EXPECT_EQ(b.Divide(a), bda);
  }
}

}  // namespace tachyon::math
