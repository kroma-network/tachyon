#include "tachyon/base/time/time_delta_flag.h"

#include <limits>

#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

namespace tachyon {
namespace base {

TEST(TimeDeltaFlagTest, ParseValue) {
  TimeDelta time_delta;
  std::string reason;
  TimeDeltaFlag time_delta_flag(&time_delta);
  EXPECT_FALSE(time_delta_flag.ParseValue("", &reason));
  EXPECT_EQ(reason, "no suffix!, please add suffix d, h, m, s, ms, us or ns");

  reason.clear();
  EXPECT_FALSE(time_delta_flag.ParseValue("1yr", &reason));
  EXPECT_EQ(reason, "no suffix!, please add suffix d, h, m, s, ms, us or ns");

  reason.clear();
  EXPECT_FALSE(time_delta_flag.ParseValue("0.5d", &reason));
  EXPECT_EQ(reason, "failed to convert to int");

  reason.clear();
  EXPECT_FALSE(time_delta_flag.ParseValue(
      absl::Substitute(
          "$0d", static_cast<int64_t>(std::numeric_limits<int>::max()) + 1),
      &reason));
  EXPECT_EQ(reason, "failed to convert to int");

  reason.clear();
  EXPECT_FALSE(time_delta_flag.ParseValue("ams", &reason));
  EXPECT_EQ(reason, "failed to convert to double");

  EXPECT_TRUE(time_delta_flag.ParseValue("1d", &reason));
  EXPECT_EQ(time_delta, Days(1));

  EXPECT_TRUE(time_delta_flag.ParseValue("1h", &reason));
  EXPECT_EQ(time_delta, Hours(1));

  EXPECT_TRUE(time_delta_flag.ParseValue("1m", &reason));
  EXPECT_EQ(time_delta, Minutes(1));

  EXPECT_TRUE(time_delta_flag.ParseValue("1s", &reason));
  EXPECT_EQ(time_delta, Seconds(1));

  EXPECT_TRUE(time_delta_flag.ParseValue("1.2s", &reason));
  EXPECT_EQ(time_delta, Seconds(1.2));

  EXPECT_TRUE(time_delta_flag.ParseValue("1ms", &reason));
  EXPECT_EQ(time_delta, Milliseconds(1));

  EXPECT_TRUE(time_delta_flag.ParseValue("1.2ms", &reason));
  EXPECT_EQ(time_delta, Milliseconds(1.2));

  EXPECT_TRUE(time_delta_flag.ParseValue("1us", &reason));
  EXPECT_EQ(time_delta, Microseconds(1));

  EXPECT_TRUE(time_delta_flag.ParseValue("1.2us", &reason));
  EXPECT_EQ(time_delta, Microseconds(1.2));

  EXPECT_TRUE(time_delta_flag.ParseValue("1ns", &reason));
  EXPECT_EQ(time_delta, Nanoseconds(1));

  EXPECT_TRUE(time_delta_flag.ParseValue("1.2ns", &reason));
  EXPECT_EQ(time_delta, Nanoseconds(1.2));
}

}  // namespace base
}  // namespace tachyon
