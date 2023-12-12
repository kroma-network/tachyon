#include "tachyon/base/strings/string_util.h"

#include "gtest/gtest.h"

namespace tachyon::base {

TEST(StringUtilTest, EmptyString) { EXPECT_EQ(EmptyString(), ""); }

TEST(StringUtilTest, StartsWith) {
  std::string_view sv = "Hello World";
  EXPECT_FALSE(StartsWith(sv, "World"));
  EXPECT_TRUE(StartsWith(sv, "Hello"));
  EXPECT_FALSE(StartsWith(sv, "Hello World!"));
  EXPECT_TRUE(StartsWith(sv, "hello", CompareCase::INSENSITIVE_ASCII));
}

TEST(StringUtilTest, EndsWith) {
  std::string_view sv = "Hello World";
  EXPECT_FALSE(EndsWith(sv, "Hello"));
  EXPECT_TRUE(EndsWith(sv, "World"));
  EXPECT_FALSE(EndsWith(sv, "!Hello World"));
  EXPECT_TRUE(EndsWith(sv, "world", CompareCase::INSENSITIVE_ASCII));
}

TEST(StringUtilTest, ConsumePrefix) {
  std::string_view sv = "Hello World";
  EXPECT_FALSE(ConsumePrefix(&sv, "World"));
  EXPECT_TRUE(ConsumePrefix(&sv, "Hello"));
  EXPECT_EQ(sv, " World");
}

TEST(StringUtilTest, ConsumeSuffix) {
  std::string_view sv = "Hello World";
  EXPECT_FALSE(ConsumeSuffix(&sv, "Hello"));
  EXPECT_TRUE(ConsumeSuffix(&sv, "World"));
  EXPECT_EQ(sv, "Hello ");
}

TEST(StringUtilTest, ConsumePrefix0x) {
  std::string_view sv = "0x1234";
  EXPECT_TRUE(ConsumePrefix0x(&sv));
  EXPECT_EQ(sv, "1234");
  sv = "3456";
  EXPECT_FALSE(ConsumePrefix0x(&sv));
  EXPECT_EQ(sv, "3456");
}

TEST(StringUtilTest, MaybePrepend0x) {
  std::string_view sv = "0x1234";
  EXPECT_EQ(MaybePrepend0x(sv), "0x1234");
  sv = "3456";
  EXPECT_EQ(MaybePrepend0x(sv), "0x3456");
}

TEST(StringUtilTest, ToHexStringWithLeadingZero) {
  std::string str = "0x1234";
  EXPECT_EQ(ToHexStringWithLeadingZero(str, 6), "0x001234");
  str = "3456";
  EXPECT_EQ(ToHexStringWithLeadingZero(str, 6), "003456");
}

}  // namespace tachyon::base
