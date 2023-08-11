#include "tachyon/base/strings/string_number_conversions.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tachyon::base {

TEST(StringNumberConversionTest, NumberToString) {
  EXPECT_EQ(NumberToString(3), "3");
  EXPECT_EQ(NumberToString(10), "10");
}

// We don't test another StringToXXX variants since they are just wrappers of
// absl::SimpleAtoi, SimpleAtof and SimpleAtod.
TEST(StringNumberConversionTest, StringToInt) {
  int output;
  EXPECT_TRUE(StringToInt("3", &output));
  EXPECT_EQ(output, 3);

  EXPECT_FALSE(StringToInt("a", &output));
}

TEST(StringNumberConversionTest, HexToString) {
  EXPECT_EQ(HexToString(3), "3");
  EXPECT_EQ(HexToString(10), "a");
}

TEST(StringNumberConversionTest, HexEncode) {
  char data[] = {1, 10, 15, 2};

  struct {
    bool use_lower;
    const char* answer;
  } tests[] = {
      {true, "010a0f02"},
      {false, "010A0F02"},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(HexEncode(data, 4, test.use_lower), test.answer);
    EXPECT_EQ(HexEncode(absl::MakeConstSpan(reinterpret_cast<uint8_t*>(data),
                                            std::size(data)),
                        test.use_lower),
              test.answer);
  }
}

// We don't test another HexStringToXXX variants since they are just wrappers of
// absl::numbers_internal::safe_strtoi_base
TEST(StringNumberConversionTest, HexStringToInt) {
  int output;
  EXPECT_TRUE(HexStringToInt("0x10", &output));
  EXPECT_EQ(output, 16);

  int output2;
  EXPECT_TRUE(HexStringToInt("10", &output2));
  EXPECT_EQ(output2, 16);
}

TEST(StringNumberConversionTest, HexStringToSomethingElse) {
  struct {
    const char* hex_input;
    std::vector<uint8_t> vec_answer;
    std::string str_answer;
    bool ret;
  } tests[] = {
      {"010a0f02", {0x1, 0xa, 0xf, 0x2}, "\x1\n\xF\x2", true},
      {"010A0F02", {0x1, 0xa, 0xf, 0x2}, "\x1\n\xF\x2", true},
      {"010x0f02", {}, "", false},
  };

  for (const auto& test : tests) {
    {
      std::vector<uint8_t> output;
      EXPECT_EQ(HexStringToBytes(test.hex_input, &output), test.ret);
      if (test.ret) {
        EXPECT_THAT(output, testing::ContainerEq(test.vec_answer));
      }
    }
    {
      std::string output;
      EXPECT_EQ(HexStringToString(test.hex_input, &output), test.ret);
      if (test.ret) {
        EXPECT_EQ(output, test.str_answer);
      }
    }
    {
      std::vector<uint8_t> output;
      output.resize(4);
      EXPECT_EQ(HexStringToSpan(test.hex_input, absl::MakeSpan(output)),
                test.ret);
      if (test.ret) {
        EXPECT_THAT(output, testing::ElementsAreArray(test.vec_answer.begin(),
                                                      test.vec_answer.end()));
      }
    }
  }
}

}  // namespace tachyon::base
