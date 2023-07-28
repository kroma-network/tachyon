// Copyright 2021 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// clang-format off

#include "tachyon/base/test/gtest_util.h"
#include "tachyon/base/time/time.h"

#include "gtest/gtest.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

namespace {

class ScopedTimebase {
 public:
  ScopedTimebase(mach_timebase_info_data_t timebase) {
    orig_timebase_ = base::TimeTicks::SetMachTimebaseInfoForTesting(timebase);
  }

  ScopedTimebase(const ScopedTimebase&) = delete;

  ScopedTimebase& operator=(const ScopedTimebase&) = delete;

  ~ScopedTimebase() {
    base::TimeTicks::SetMachTimebaseInfoForTesting(orig_timebase_);
  }

 private:
  mach_timebase_info_data_t orig_timebase_;
};

mach_timebase_info_data_t kIntelTimebase = {1, 1};

// A sample (not definitive) timebase for M1.
mach_timebase_info_data_t kM1Timebase = {125, 3};

}  // namespace

namespace tachyon::base {
namespace {

base::Time NoonOnDate(int year, int month, int day) {
  base::Time::Exploded exploded;
  exploded.year = year;
  exploded.month = month;
  exploded.day_of_week = 0;  // Not correct, but FromExploded permits it
  exploded.day_of_month = day;
  exploded.hour = 12;
  exploded.minute = 0;
  exploded.second = 0;
  exploded.millisecond = 0;
  base::Time imploded;
  CHECK(base::Time::FromUTCExploded(exploded, &imploded));
  return imploded;
}

void CheckRoundTrip(int y, int m, int d) {
  base::Time original = NoonOnDate(y, m, d);
  base::Time roundtrip = Time::FromNSDate(original.ToNSDate());
  EXPECT_EQ(original, roundtrip);
}

TEST(TimeMacTest, RoundTripNSDate) {
  CheckRoundTrip(1911, 12, 14);
  CheckRoundTrip(1924, 9, 28);
  CheckRoundTrip(1926, 5, 12);
  CheckRoundTrip(1969, 7, 24);
}

TEST(TimeMacTest, MachTimeToMicrosecondsIntelTimebase) {
  ScopedTimebase timebase(kIntelTimebase);

  // Perform the conversion.
  uint64_t kArbitraryTicks = 59090101000;
  TimeDelta result = TimeDelta::FromMachTime(kArbitraryTicks);

  // With Intel the output should be the input.
  EXPECT_EQ(Nanoseconds(kArbitraryTicks), result);
}

TEST(TimeMacTest, MachTimeToMicrosecondsM1Timebase) {
  ScopedTimebase timebase(kM1Timebase);

  // Use a tick count that's divisible by 3.
  const uint64_t kArbitraryTicks = 92738127000;
  TimeDelta result = TimeDelta::FromMachTime(kArbitraryTicks);

  const uint64_t kExpectedResult =
      kArbitraryTicks * kM1Timebase.numer / kM1Timebase.denom;
  EXPECT_EQ(Nanoseconds(kExpectedResult), result);
}

// Tests MachTimeToMicroseconds when
// mach_timebase_info_data_t.numer and mach_timebase_info_data_t.denom
// are equal.
TEST(TimeMacTest, MachTimeToMicrosecondsEqualTimebaseMembers) {
  // These members would produce overflow but don't because
  // MachTimeToMicroseconds should skip the timebase conversion
  // when they're equal.
  ScopedTimebase timebase({UINT_MAX, UINT_MAX});

  uint64_t kArbitraryTicks = 175920053729;
  TimeDelta result = TimeDelta::FromMachTime(kArbitraryTicks);

  // With a unity timebase the output should be the input.
  EXPECT_EQ(Nanoseconds(kArbitraryTicks), result);
}

TEST(TimeMacTest, MachTimeToMicrosecondsOverflowDetection) {
  const uint32_t kArbitraryNumer = 1234567;
  ScopedTimebase timebase({kArbitraryNumer, 1});

  // Expect an overflow.
  EXPECT_CHECK_DEATH(
      TimeDelta::FromMachTime(std::numeric_limits<uint64_t>::max()));
}

// Tests that there's no overflow in MachTimeToMicroseconds even with
// std::numeric_limits<uint64_t>::max() ticks on Intel.
TEST(TimeMacTest, MachTimeToMicrosecondsNoOverflowIntel) {
  ScopedTimebase timebase(kIntelTimebase);

  // The incoming Mach time ticks are on the order of nanoseconds while the
  // return result is microseconds. Even though we're passing in the largest
  // tick count the result should be orders of magnitude smaller. On Intel the
  // mapping from ticks to nanoseconds is 1:1 so we wouldn't ever expect an
  // overflow when applying the timebase conversion.
  TimeDelta::FromMachTime(std::numeric_limits<uint64_t>::max());
}

// Tests that there's no overflow in MachTimeToMicroseconds even with
// std::numeric_limits<uint64_t>::max() ticks on M1.
TEST(TimeMacTest, MachTimeToMicrosecondsNoOverflowM1) {
  ScopedTimebase timebase(kM1Timebase);

  // The incoming Mach time ticks are on the order of nanoseconds while the
  // return result is microseconds. Even though we're passing in the largest
  // tick count the result should be orders of magnitude smaller. Expect that
  // FromMachTime(), when applying the timebase conversion, is smart enough to
  // not multiply first and generate an overflow.
  TimeDelta::FromMachTime(std::numeric_limits<uint64_t>::max());
}

// Tests that there's no underflow in MachTimeToMicroseconds on Intel.
TEST(TimeMacTest, MachTimeToMicrosecondsNoUnderflowIntel) {
  ScopedTimebase timebase(kIntelTimebase);

  // On Intel the timebase conversion is 1:1, so min ticks is one microsecond
  // worth of nanoseconds.
  const uint64_t kMinimumTicks = base::Time::kNanosecondsPerMicrosecond;
  const uint64_t kOneMicrosecond = 1;
  EXPECT_EQ(kOneMicrosecond,
            TimeDelta::FromMachTime(kMinimumTicks).InMicroseconds() * 1UL);

  // If we have even one fewer tick (i.e. not enough ticks to constitute a full
  // microsecond) the integer rounding should result in 0 microseconds.
  const uint64_t kZeroMicroseconds = 0;
  EXPECT_EQ(kZeroMicroseconds,
            TimeDelta::FromMachTime(kMinimumTicks - 1).InMicroseconds() * 1UL);
}

// Tests that there's no underflow in MachTimeToMicroseconds for M1.
TEST(TimeMacTest, MachTimeToMicrosecondsNoUnderflowM1) {
  ScopedTimebase timebase(kM1Timebase);

  // Microseconds is mach_time multiplied by kM1Timebase.numer /
  // (kM1Timebase.denom * base::Time::kNanosecondsPerMicrosecond). Inverting
  // that should be the minimum number of ticks to get a single microsecond in
  // return. If we get zero it means an underflow in the conversion. For example
  // if FromMachTime() first divides mach_time by kM1Timebase.denom *
  // base::Time::kNanosecondsPerMicrosecond we'll get zero back.
  const uint64_t kMinimumTicks =
      (kM1Timebase.denom * base::Time::kNanosecondsPerMicrosecond) /
      kM1Timebase.numer;
  const uint64_t kOneMicrosecond = 1;
  EXPECT_EQ(kOneMicrosecond,
            TimeDelta::FromMachTime(kMinimumTicks).InMicroseconds() * 1UL);

  // If we have even one fewer tick (i.e. not enough ticks to constitute a full
  // microsecond) the integer rounding should result in 0 microseconds.
  const uint64_t kZeroMicroseconds = 0;
  EXPECT_EQ(kZeroMicroseconds,
            TimeDelta::FromMachTime(kMinimumTicks - 1).InMicroseconds() * 1UL);
}

}  // namespace
}  // namespace tachyon::base

// clang-format on