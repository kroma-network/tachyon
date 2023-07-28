// Copyright (c) 2020 The Console Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/flag/flag.h"

#include "gtest/gtest.h"

namespace tachyon::base {

TEST(FlagTest, ShortName) {
  bool value;
  BoolFlag flag(&value);
  flag.set_short_name("-a");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("--a");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("a");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("-ab");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("-1");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("-_");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("-b");
  EXPECT_EQ(flag.short_name(), "-b");
}

TEST(FlagTest, LongName) {
  bool value;
  BoolFlag flag(&value);
  flag.set_long_name("--a");
  EXPECT_EQ(flag.long_name(), "--a");
  flag.set_long_name("-a");
  EXPECT_EQ(flag.long_name(), "--a");
  flag.set_long_name("a");
  EXPECT_EQ(flag.long_name(), "--a");
  flag.set_long_name("--1");
  EXPECT_EQ(flag.long_name(), "--a");
  flag.set_long_name("--_");
  EXPECT_EQ(flag.long_name(), "--a");
  flag.set_long_name("--a_");
  EXPECT_EQ(flag.long_name(), "--a_");
}

TEST(FlagTest, Name) {
  bool value;
  BoolFlag flag(&value);
  flag.set_name("a");
  EXPECT_EQ(flag.name(), "a");
  flag.set_name("-a");
  EXPECT_EQ(flag.name(), "a");
  flag.set_name("--a");
  EXPECT_EQ(flag.name(), "a");
  flag.set_name("1");
  EXPECT_EQ(flag.name(), "a");
  flag.set_name("_");
  EXPECT_EQ(flag.name(), "a");
  flag.set_name("a_");
  EXPECT_EQ(flag.name(), "a_");
}

TEST(FlagTest, ParseValue) {
  bool bool_value = false;
  std::string reason;
  BoolFlag bool_flag(&bool_value);
  EXPECT_TRUE(bool_flag.ParseValue("", &reason));
  EXPECT_EQ(reason, "");
  EXPECT_TRUE(bool_value);

  reason.clear();
  int16_t int16_value;
  Int16Flag int16_flag(&int16_value);
  EXPECT_TRUE(int16_flag.ParseValue("123", &reason));
  EXPECT_EQ(reason, "");
  EXPECT_EQ(int16_value, 123);
  EXPECT_FALSE(int16_flag.ParseValue("a", &reason));
  EXPECT_EQ(reason, "failed to convert int (\"a\")");
  EXPECT_FALSE(int16_flag.ParseValue("40000", &reason));
  EXPECT_EQ(reason, "40000 is out of its range");
  EXPECT_EQ(int16_value, 123);

  reason.clear();
  std::string string_value;
  StringFlag string_flag(&string_value);
  EXPECT_TRUE(string_flag.ParseValue("abc", &reason));
  EXPECT_EQ(reason, "");
  EXPECT_EQ(string_value, "abc");
  EXPECT_FALSE(string_flag.ParseValue("", &reason));
  EXPECT_EQ(reason, "input is empty");
  EXPECT_EQ(string_value, "abc");

  reason.clear();
  std::string choice_value;
  StringChoicesFlag choices_flag(
      &choice_value, std::vector<std::string>{"cat", "dog", "duck"});
  EXPECT_TRUE(choices_flag.ParseValue("cat", &reason));
  EXPECT_EQ(reason, "");
  EXPECT_TRUE(choices_flag.ParseValue("dog", &reason));
  EXPECT_EQ(reason, "");
  EXPECT_TRUE(choices_flag.ParseValue("duck", &reason));
  EXPECT_EQ(reason, "");
  EXPECT_FALSE(choices_flag.ParseValue("bird", &reason));
  EXPECT_EQ(reason, "bird is not in choices");

  reason.clear();
  int32_t int32_value;
  Int32RangeFlag int32_range_flag(&int32_value, 1, 5);
  int32_range_flag.set_less_than_or_equal_to(true);
  EXPECT_TRUE(int32_range_flag.ParseValue("2", &reason));
  EXPECT_EQ(reason, "");
  EXPECT_TRUE(int32_range_flag.ParseValue("5", &reason));
  EXPECT_EQ(reason, "");
  EXPECT_FALSE(int32_range_flag.ParseValue("1", &reason));
  EXPECT_EQ(reason, "1 is not in range");
  EXPECT_FALSE(int32_range_flag.ParseValue("6", &reason));
  EXPECT_EQ(reason, "6 is not in range");
}

}  // namespace tachyon::base
