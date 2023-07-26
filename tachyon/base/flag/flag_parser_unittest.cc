// Copyright (c) 2020 The Console Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/flag/flag_parser.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/test/scoped_environment.h"

namespace tachyon {
namespace base {

#define EXPECT_PARSE_TRUE(...)        \
  const char* argv[] = {__VA_ARGS__}; \
  std::string error;                  \
  EXPECT_TRUE(parser.Parse(std::size(argv), const_cast<char**>(argv), &error))

#define EXPECT_PARSE_FALSE(...)       \
  const char* argv[] = {__VA_ARGS__}; \
  std::string error;                  \
  EXPECT_FALSE(parser.Parse(std::size(argv), const_cast<char**>(argv), &error))

TEST(FlagParserTest, ValidateInternally) {
  {
    FlagParser parser;
    uint16_t value;
    parser.AddFlag<Uint16Flag>(&value);
    std::string error;
    EXPECT_FALSE(parser.ValidateInternally(&error));
    EXPECT_EQ(error, "Flag should be positional or optional.");
  }
  {
    FlagParser parser;
    uint16_t value;
    parser.AddFlag<Uint16Flag>(&value).set_name("value").set_short_name("-v");
    std::string error;
    EXPECT_FALSE(parser.ValidateInternally(&error));
    EXPECT_EQ(error,
              "\"value\" is positional and optional, please choose either one "
              "of them.");
  }
  {
    FlagParser parser;
    parser.AddSubParser().set_short_name("-a");
    std::string error;
    EXPECT_FALSE(parser.ValidateInternally(&error));
    EXPECT_EQ(error, "Subparser \"-a\" should be positional.");
  }
  {
    FlagParser parser;
    uint16_t value;
    parser.AddFlag<Uint16Flag>(&value).set_short_name("-v");
    parser.AddFlag<Uint16Flag>(&value).set_name("value");
    std::string error;
    EXPECT_FALSE(parser.ValidateInternally(&error));
    EXPECT_EQ(error, "\"value\" should be before any optional arguments.");
  }
  {
    FlagParser parser;
    bool value;
    parser.AddFlag<BoolFlag>(&value).set_name("value");
    std::string error;
    EXPECT_FALSE(parser.ValidateInternally(&error));
    EXPECT_EQ(error,
              "\"value\" can't parse a value, how about considering using "
              "set_short_name() or set_long_name()?");
  }
  {
    FlagParser parser;
    uint16_t value;
    parser.AddSubParser().set_name("a");
    parser.AddFlag<Uint16Flag>(&value).set_name("value");
    std::string error;
    EXPECT_FALSE(parser.ValidateInternally(&error));
    EXPECT_EQ(error,
              "\"value\" can't be positional if the parser has "
              "subparser, how about considering using "
              "set_short_name() or set_long_name()?");
  }
  {
    FlagParser parser;
    uint16_t value;
    parser.AddFlag<Uint16Flag>(&value).set_short_name("-a");
    SubParser& sub_parser = parser.AddSubParser();
    sub_parser.set_name("test");
    sub_parser.AddFlag<Uint16Flag>(&value).set_name("name");
    std::string error;
    EXPECT_FALSE(parser.ValidateInternally(&error));
    EXPECT_EQ(error, "SubParser should be at the very front.");
  }
}

TEST(FlagParserTest, UndefinedArgument) {
  FlagParser parser;
  uint16_t value;
  parser.AddFlag<Uint16Flag>(&value).set_long_name("--value");
  {
    EXPECT_PARSE_FALSE("program", "--v", "16");
    EXPECT_EQ(error, "met unknown argument: \"--v\".");
  }
  {
    EXPECT_PARSE_FALSE("program", "--val", "16");
    EXPECT_EQ(error,
              "met unknown argument: \"--val\", maybe you mean \"--value\"?");
  }
}

TEST(FlagParserTest, DefaultValue) {
  FlagParser parser;
  uint16_t value;
  parser.AddFlag<Uint16Flag>(&value)
      .set_default_value(static_cast<uint16_t>(12))
      .set_short_name("-v");
  {
    EXPECT_PARSE_TRUE("program");
    EXPECT_EQ(value, 12);
  }
}

TEST(FlagParserTest, PositionalArgumtens) {
  FlagParser parser;
  uint16_t value;
  uint16_t value2;
  parser.AddFlag<Uint16Flag>(&value).set_name("flag");
  parser.AddFlag<Uint16Flag>(&value2).set_name("flag2");
  {
    EXPECT_PARSE_FALSE("program", "12");
  }
  { EXPECT_PARSE_TRUE("program", "12", "34"); }
  EXPECT_EQ(value, 12);
  EXPECT_EQ(value2, 34);
}

TEST(FlagParserTest, RequiredOptionalArguments) {
  FlagParser parser;
  uint16_t value;
  uint16_t value2;
  parser.AddFlag<Uint16Flag>(&value).set_short_name("-a");
  parser.AddFlag<Uint16Flag>(&value2).set_short_name("-b").set_required();
  {
    EXPECT_PARSE_FALSE("program", "-a", "12");
    EXPECT_EQ(error, "\"-b\" is required, but not set.");
  }
  { EXPECT_PARSE_TRUE("program", "-b", "34"); }
  EXPECT_EQ(value2, 34);
  { EXPECT_PARSE_TRUE("program", "-a", "56", "-b", "78"); }
  EXPECT_EQ(value, 56);
  EXPECT_EQ(value2, 78);
}

TEST(FlagParserTest, ConcatenatedOptionalFlags) {
  FlagParser parser;
  bool value = false;
  bool value2 = false;
  int32_t value3;
  parser.AddFlag<BoolFlag>(&value).set_short_name("-a");
  parser.AddFlag<BoolFlag>(&value2).set_short_name("-b");
  parser.AddFlag<Int32Flag>(&value3).set_short_name("-c");
  { EXPECT_PARSE_TRUE("program", "-ab"); }
  EXPECT_TRUE(value);
  EXPECT_TRUE(value2);
  {
    EXPECT_PARSE_FALSE("program", "-ac");
    EXPECT_EQ(error, "met unknown argument: \"-ac\", maybe you mean \"-a\"?");
  }
}

TEST(FlagParserTest, VectorFlag) {
  FlagParser parser;
  std::vector<int> numbers;
  parser.AddFlag<Flag<std::vector<int>>>(&numbers).set_short_name("-a");
  {
    EXPECT_PARSE_TRUE("program", "-a", "1", "-a", "2", "-a", "3");
    EXPECT_THAT(numbers, testing::ElementsAre(1, 2, 3));
  }
}

TEST(FlagParserTest, CustomParseValueCallback) {
  FlagParser parser;
  std::string value;
  parser
      .AddFlag<StringFlag>([&value](std::string_view arg, std::string* reason) {
        if (arg == "cat" || arg == "dog") {
          value = std::string(arg);
          return true;
        }
        *reason = absl::Substitute("$0 is not either cat or dog", arg);
        return false;
      })
      .set_short_name("-a");
  {
    EXPECT_PARSE_FALSE("program", "-a", "pig");
    EXPECT_EQ(
        error,
        "\"-a\" is failed to parse: (reason: pig is not either cat or dog).");
  }
  {
    EXPECT_PARSE_TRUE("program", "-a", "cat");
    EXPECT_EQ(value, "cat");
  }
}

TEST(FlagParserTest, ParseValueFromEnvironment) {
  FlagParser parser;
  std::string value;
  parser.AddFlag<StringFlag>(&value).set_env_name("VALUE").set_short_name("-v");
  {
    ScopedEnvironment env("VALUE", "value");
    EXPECT_PARSE_TRUE("program");
    EXPECT_EQ(value, "value");
  }
}

TEST(FlagParserTest, ParseKnown) {
  FlagParser parser;
  int value;
  parser.AddFlag<IntFlag>(&value).set_short_name("-a");
  bool value2;
  parser.AddFlag<BoolFlag>(&value2).set_short_name("-b");
  {
    value = 0;
    value2 = false;
    const char* argv[] = {"program", "-a", "1", "--unknown", "-b"};
    int argc = std::size(argv);
    std::string error;
    EXPECT_TRUE(parser.ParseKnown(&argc, const_cast<char**>(argv), &error));
    EXPECT_EQ(value, 1);
    EXPECT_EQ(value2, true);
    EXPECT_EQ(argc, 2);
    EXPECT_STREQ(argv[0], "program");
    EXPECT_STREQ(argv[1], "--unknown");
  }

  {
    value = 0;
    value2 = false;
    const char* argv[] = {"program", "-a", "1", "-b", "--unknown", "2"};
    int argc = std::size(argv);
    std::string error;
    EXPECT_TRUE(parser.ParseKnown(&argc, const_cast<char**>(argv), &error));
    EXPECT_EQ(value, 1);
    EXPECT_EQ(value2, true);
    EXPECT_EQ(argc, 3);
    EXPECT_STREQ(argv[0], "program");
    EXPECT_STREQ(argv[1], "--unknown");
    EXPECT_STREQ(argv[2], "2");
  }
}

TEST(FlagParserTest, ParseWithForward) {
  FlagParser parser;
  int value;
  parser.AddFlag<IntFlag>(&value).set_short_name("-a");
  bool value2;
  parser.AddFlag<BoolFlag>(&value2).set_short_name("-b");
  {
    value = 0;
    value2 = false;
    const char* argv[] = {"program", "-a", "1", "--", "-b"};
    int argc = std::size(argv);
    std::vector<std::string> forward;
    std::string error;
    EXPECT_TRUE(parser.ParseWithForward(argc, const_cast<char**>(argv),
                                        &forward, &error));
    EXPECT_EQ(value, 1);
    EXPECT_EQ(value2, false);
    EXPECT_THAT(forward, testing::ElementsAre("-b"));
  }
}

TEST(FlagParserTest, SubParserTest) {
  FlagParser parser;
  int a;
  int b;
  bool verbose = false;
  SubParser& add_parser = parser.AddSubParser().set_name("add");
  add_parser.AddFlag<Int32Flag>(&a).set_name("a");
  add_parser.AddFlag<Int32Flag>(&b).set_name("b");
  SubParser& sub_parser = parser.AddSubParser().set_name("sub");
  sub_parser.AddFlag<Int32Flag>(&a).set_name("a");
  sub_parser.AddFlag<Int32Flag>(&b).set_name("b");
  parser.AddFlag<BoolFlag>(&verbose).set_short_name("-v");
  {
    EXPECT_PARSE_TRUE("program", "add", "1", "2");
    EXPECT_TRUE(add_parser.is_set());
    EXPECT_FALSE(sub_parser.is_set());
    EXPECT_EQ(1, a);
    EXPECT_EQ(2, b);
    EXPECT_FALSE(verbose);
  }

  add_parser.reset();
  {
    EXPECT_PARSE_TRUE("program", "-v", "add", "1", "2");
    EXPECT_TRUE(add_parser.is_set());
    EXPECT_FALSE(sub_parser.is_set());
    EXPECT_EQ(1, a);
    EXPECT_EQ(2, b);
    EXPECT_TRUE(verbose);
  }
}

#undef EXPECT_PARSE_TRUE
#undef EXPECT_PARSE_FALSE

}  // namespace base
}  // namespace tachyon
