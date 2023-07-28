#include "tachyon/base/environment.h"

#include "gtest/gtest.h"

namespace tachyon::base {

TEST(Environment, BasicTest) {
  std::string_view value;
  EXPECT_FALSE(Environment::Has("foo"));
  EXPECT_FALSE(Environment::Get("foo", &value));

  EXPECT_TRUE(Environment::Set("foo", "bar"));
  EXPECT_TRUE(Environment::Has("foo"));
  EXPECT_TRUE(Environment::Get("foo", &value));
  EXPECT_STREQ(value.data(), "bar");

  EXPECT_TRUE(Environment::Unset("foo"));
  EXPECT_FALSE(Environment::Has("foo"));
}

}  // namespace tachyon::base
