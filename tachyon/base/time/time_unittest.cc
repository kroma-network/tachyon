// Copyright (c) 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/time/time.h"

#include "gtest/gtest-death-test.h"
#include "gtest/gtest.h"

#include "tachyon/base/threading/platform_thread.h"

namespace tachyon {
namespace base {

TEST(TimeTest, Max) {
  constexpr Time kMax = Time::Max();
  static_assert(kMax.is_max());
  static_assert(kMax == Time::Max());
  EXPECT_GT(kMax, Time::Now());
  static_assert(kMax > Time());
  EXPECT_TRUE((Time::Now() - kMax).is_negative());
  EXPECT_TRUE((kMax - Time::Now()).is_positive());
}

TEST(TimeTest, MaxConversions) {
  Time t = Time::FromDoubleT(std::numeric_limits<double>::infinity());
  EXPECT_TRUE(t.is_max());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), t.ToDoubleT());

  t = Time::FromTimeT(std::numeric_limits<time_t>::max());
  EXPECT_TRUE(t.is_max());
  EXPECT_EQ(std::numeric_limits<time_t>::max(), t.ToTimeT());

#if BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
  struct timeval tval;
  tval.tv_sec = std::numeric_limits<time_t>::max();
  tval.tv_usec = static_cast<suseconds_t>(Time::kMicrosecondsPerSecond) - 1;
  t = Time::FromTimeVal(tval);
  EXPECT_TRUE(t.is_max());
  tval = t.ToTimeVal();
  EXPECT_EQ(std::numeric_limits<time_t>::max(), tval.tv_sec);
  EXPECT_EQ(static_cast<suseconds_t>(Time::kMicrosecondsPerSecond) - 1,
            tval.tv_usec);
#endif

#if BUILDFLAG(IS_APPLE)
  t = Time::FromCFAbsoluteTime(std::numeric_limits<CFAbsoluteTime>::infinity());
  EXPECT_TRUE(t.is_max());
  EXPECT_EQ(std::numeric_limits<CFAbsoluteTime>::infinity(),
            t.ToCFAbsoluteTime());
#endif

#if BUILDFLAG(IS_WIN)
  FILETIME ftime;
  ftime.dwHighDateTime = std::numeric_limits<DWORD>::max();
  ftime.dwLowDateTime = std::numeric_limits<DWORD>::max();
  t = Time::FromFileTime(ftime);
  EXPECT_TRUE(t.is_max());
  ftime = t.ToFileTime();
  EXPECT_EQ(std::numeric_limits<DWORD>::max(), ftime.dwHighDateTime);
  EXPECT_EQ(std::numeric_limits<DWORD>::max(), ftime.dwLowDateTime);
#endif
}

TEST(TimeTest, Min) {
  constexpr Time kMin = Time::Min();
  static_assert(kMin.is_min());
  static_assert(kMin == Time::Min());
  EXPECT_LT(kMin, Time::Now());
  static_assert(kMin < Time());
  EXPECT_TRUE((Time::Now() - kMin).is_positive());
  EXPECT_TRUE((kMin - Time::Now()).is_negative());
}

TEST(TimeTicks, Deltas) {
  for (int index = 0; index < 50; index++) {
    TimeTicks ticks_start = TimeTicks::Now();
    PlatformThread::Sleep(Milliseconds(10));
    TimeTicks ticks_stop = TimeTicks::Now();
    TimeDelta delta = ticks_stop - ticks_start;
    // Note:  Although we asked for a 10ms sleep, if the
    // time clock has a finer granularity than the Sleep()
    // clock, it is quite possible to wakeup early.  Here
    // is how that works:
    //      Time(ms timer)      Time(us timer)
    //          5                   5010
    //          6                   6010
    //          7                   7010
    //          8                   8010
    //          9                   9000
    // Elapsed  4ms                 3990us
    //
    // Unfortunately, our InMilliseconds() function truncates
    // rather than rounds.  We should consider fixing this
    // so that our averages come out better.
    EXPECT_GE(delta.InMilliseconds(), 9);
    EXPECT_GE(delta.InMicroseconds(), 9000);
    EXPECT_EQ(delta.InSeconds(), 0);
  }
}

TEST(ThreadTicks, ThreadNow) {
  if (ThreadTicks::IsSupported()) {
    TimeTicks begin = TimeTicks::Now();
    ThreadTicks begin_thread = ThreadTicks::Now();
    // Make sure that ThreadNow value is non-zero.
    EXPECT_GT(begin_thread, ThreadTicks());
    // Sleep for 10 milliseconds to get the thread de-scheduled.
    PlatformThread::Sleep(Milliseconds(10));
    ThreadTicks end_thread = ThreadTicks::Now();
    TimeTicks end = TimeTicks::Now();
    TimeDelta delta = end - begin;
    TimeDelta delta_thread = end_thread - begin_thread;
    // Make sure that some thread time have elapsed.
    EXPECT_GE(delta_thread.InMicroseconds(), 0);
    // But the thread time is at least 9ms less than clock time.
    TimeDelta difference = delta - delta_thread;
    EXPECT_GE(difference.InMicroseconds(), 9000);
  }
}

TEST(TimeDelta, FromAndIn) {
  // static_assert also checks that the contained expression is a constant
  // expression, meaning all its components are suitable for initializing global
  // variables.
  static_assert(Days(2) == Hours(48));
  static_assert(Hours(3) == Minutes(180));
  static_assert(Minutes(2) == Seconds(120));
  static_assert(Seconds(2) == Milliseconds(2000));
  static_assert(Milliseconds(2) == Microseconds(2000));
  static_assert(Seconds(2.3) == Milliseconds(2300));
  static_assert(Milliseconds(2.5) == Microseconds(2500));
  EXPECT_EQ(Days(13).InDays(), 13);
  static_assert(Hours(13).InHours() == 13);
  static_assert(Minutes(13).InMinutes() == 13);
  static_assert(Seconds(13).InSeconds() == 13);
  static_assert(Seconds(13).InSecondsF() == 13.0);
  EXPECT_EQ(Milliseconds(13).InMilliseconds(), 13);
  EXPECT_EQ(Milliseconds(13).InMillisecondsF(), 13.0);
  static_assert(Seconds(13.1).InSeconds() == 13);
  static_assert(Seconds(13.1).InSecondsF() == 13.1);
  EXPECT_EQ(Milliseconds(13.3).InMilliseconds(), 13);
  EXPECT_EQ(Milliseconds(13.3).InMillisecondsF(), 13.3);
  static_assert(Microseconds(13).InMicroseconds() == 13);
  static_assert(Microseconds(13.3).InMicroseconds() == 13);
  EXPECT_EQ(Milliseconds(3.45678).InMillisecondsF(), 3.456);
  static_assert(Nanoseconds(12345).InNanoseconds() == 12000);
  static_assert(Nanoseconds(12345.678).InNanoseconds() == 12000);
}

TEST(TimeDelta, InRoundsTowardsZero) {
  EXPECT_EQ(Hours(23).InDays(), 0);
  EXPECT_EQ(Hours(-23).InDays(), 0);
  static_assert(Minutes(59).InHours() == 0);
  static_assert(Minutes(-59).InHours() == 0);
  static_assert(Seconds(59).InMinutes() == 0);
  static_assert(Seconds(-59).InMinutes() == 0);
  static_assert(Milliseconds(999).InSeconds() == 0);
  static_assert(Milliseconds(-999).InSeconds() == 0);
  EXPECT_EQ(Microseconds(999).InMilliseconds(), 0);
  EXPECT_EQ(Microseconds(-999).InMilliseconds(), 0);
}

TEST(TimeDelta, InDaysFloored) {
  EXPECT_EQ(Hours(-25).InDaysFloored(), -2);
  EXPECT_EQ(Hours(-24).InDaysFloored(), -1);
  EXPECT_EQ(Hours(-23).InDaysFloored(), -1);

  EXPECT_EQ(Hours(-1).InDaysFloored(), -1);
  EXPECT_EQ(Hours(0).InDaysFloored(), 0);
  EXPECT_EQ(Hours(1).InDaysFloored(), 0);

  EXPECT_EQ(Hours(23).InDaysFloored(), 0);
  EXPECT_EQ(Hours(24).InDaysFloored(), 1);
  EXPECT_EQ(Hours(25).InDaysFloored(), 1);
}

TEST(TimeDelta, InSecondsFloored) {
  EXPECT_EQ(Seconds(13.1).InSecondsFloored(), 13);
  EXPECT_EQ(Seconds(13.9).InSecondsFloored(), 13);
  EXPECT_EQ(Seconds(13).InSecondsFloored(), 13);

  EXPECT_EQ(Milliseconds(1001).InSecondsFloored(), 1);
  EXPECT_EQ(Milliseconds(1000).InSecondsFloored(), 1);
  EXPECT_EQ(Milliseconds(999).InSecondsFloored(), 0);
  EXPECT_EQ(Milliseconds(1).InSecondsFloored(), 0);
  EXPECT_EQ(Milliseconds(0).InSecondsFloored(), 0);
  EXPECT_EQ(Milliseconds(-1).InSecondsFloored(), -1);
  EXPECT_EQ(Milliseconds(-1000).InSecondsFloored(), -1);
  EXPECT_EQ(Milliseconds(-1001).InSecondsFloored(), -2);
}

TEST(TimeDelta, InMillisecondsRoundedUp) {
  EXPECT_EQ(Microseconds(-1001).InMillisecondsRoundedUp(), -1);
  EXPECT_EQ(Microseconds(-1000).InMillisecondsRoundedUp(), -1);
  EXPECT_EQ(Microseconds(-999).InMillisecondsRoundedUp(), 0);

  EXPECT_EQ(Microseconds(-1).InMillisecondsRoundedUp(), 0);
  EXPECT_EQ(Microseconds(0).InMillisecondsRoundedUp(), 0);
  EXPECT_EQ(Microseconds(1).InMillisecondsRoundedUp(), 1);

  EXPECT_EQ(Microseconds(999).InMillisecondsRoundedUp(), 1);
  EXPECT_EQ(Microseconds(1000).InMillisecondsRoundedUp(), 1);
  EXPECT_EQ(Microseconds(1001).InMillisecondsRoundedUp(), 2);
}

// Check that near-min/max values saturate rather than overflow when converted
// lossily with InXXX() functions.  Only integral hour, minute, and nanosecond
// conversions are checked, since those are the only cases where the return type
// is small enough for saturation or overflow to occur.
TEST(TimeDelta, InXXXOverflow) {
  constexpr TimeDelta kLargeDelta =
      Microseconds(std::numeric_limits<int64_t>::max() - 1);
  static_assert(!kLargeDelta.is_max());
  static_assert(std::numeric_limits<int>::max() == kLargeDelta.InHours());
  static_assert(std::numeric_limits<int>::max() == kLargeDelta.InMinutes());
  static_assert(std::numeric_limits<int64_t>::max() ==
                kLargeDelta.InNanoseconds());

  constexpr TimeDelta kLargeNegative =
      Microseconds(std::numeric_limits<int64_t>::min() + 1);
  static_assert(!kLargeNegative.is_min());
  static_assert(std::numeric_limits<int>::min() == kLargeNegative.InHours());
  static_assert(std::numeric_limits<int>::min() == kLargeNegative.InMinutes());
  static_assert(std::numeric_limits<int64_t>::min() ==
                kLargeNegative.InNanoseconds());
}

#if BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
TEST(TimeDelta, TimeSpecConversion) {
  TimeDelta delta = Seconds(0);
  struct timespec result = delta.ToTimeSpec();
  EXPECT_EQ(result.tv_sec, 0);
  EXPECT_EQ(result.tv_nsec, 0);
  EXPECT_EQ(delta, TimeDelta::FromTimeSpec(result));

  delta = Seconds(1);
  result = delta.ToTimeSpec();
  EXPECT_EQ(result.tv_sec, 1);
  EXPECT_EQ(result.tv_nsec, 0);
  EXPECT_EQ(delta, TimeDelta::FromTimeSpec(result));

  delta = Microseconds(1);
  result = delta.ToTimeSpec();
  EXPECT_EQ(result.tv_sec, 0);
  EXPECT_EQ(result.tv_nsec, 1000);
  EXPECT_EQ(delta, TimeDelta::FromTimeSpec(result));

  delta = Microseconds(Time::kMicrosecondsPerSecond + 1);
  result = delta.ToTimeSpec();
  EXPECT_EQ(result.tv_sec, 1);
  EXPECT_EQ(result.tv_nsec, 1000);
  EXPECT_EQ(delta, TimeDelta::FromTimeSpec(result));
}
#endif  // BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)

TEST(TimeDelta, Hz) {
  static_assert(Hertz(1) == Seconds(1));
  EXPECT_EQ(Hertz(0), TimeDelta::Max());
  static_assert(Hertz(-1) == Seconds(-1));
  static_assert(Hertz(1000) == Milliseconds(1));
  static_assert(Hertz(0.5) == Seconds(2));
  static_assert(Hertz(std::numeric_limits<double>::infinity()) == TimeDelta());

  static_assert(Seconds(1).ToHz() == 1);
  static_assert(TimeDelta::Max().ToHz() == 0);
  static_assert(Seconds(-1).ToHz() == -1);
  static_assert(Milliseconds(1).ToHz() == 1000);
  static_assert(Seconds(2).ToHz() == 0.5);
  EXPECT_EQ(TimeDelta().ToHz(), std::numeric_limits<double>::infinity());

  // 60 Hz can't be represented exactly.
  static_assert(Hertz(60) * 60 != Seconds(1));
  static_assert(Hertz(60).ToHz() != 60);
  EXPECT_EQ(ClampRound(Hertz(60).ToHz()), 60);
}

// We could define this separately for Time, TimeTicks and TimeDelta but the
// definitions would be identical anyway.
template <class Any>
std::string AnyToString(Any any) {
  std::ostringstream oss;
  oss << any;
  return oss.str();
}

TEST(TimeDelta, Magnitude) {
  constexpr int64_t zero = 0;
  static_assert(Microseconds(zero) == Microseconds(zero).magnitude());

  constexpr int64_t one = 1;
  constexpr int64_t negative_one = -1;
  static_assert(Microseconds(one) == Microseconds(one).magnitude());
  static_assert(Microseconds(one) == Microseconds(negative_one).magnitude());

  constexpr int64_t max_int64_minus_one =
      std::numeric_limits<int64_t>::max() - 1;
  constexpr int64_t min_int64_plus_two =
      std::numeric_limits<int64_t>::min() + 2;
  static_assert(Microseconds(max_int64_minus_one) ==
                Microseconds(max_int64_minus_one).magnitude());
  static_assert(Microseconds(max_int64_minus_one) ==
                Microseconds(min_int64_plus_two).magnitude());

  static_assert(TimeDelta::Max() == TimeDelta::Min().magnitude());
}

TEST(TimeDelta, ZeroMinMax) {
  constexpr TimeDelta kZero;
  static_assert(kZero.is_zero());

  constexpr TimeDelta kMax = TimeDelta::Max();
  static_assert(kMax.is_max());
  static_assert(kMax == TimeDelta::Max());
  static_assert(kMax > Days(100 * 365));
  static_assert(kMax > kZero);

  constexpr TimeDelta kMin = TimeDelta::Min();
  static_assert(kMin.is_min());
  static_assert(kMin == TimeDelta::Min());
  static_assert(kMin < Days(-100 * 365));
  static_assert(kMin < kZero);
}

TEST(TimeDelta, MaxConversions) {
  // static_assert also confirms constexpr works as intended.
  constexpr TimeDelta kMax = TimeDelta::Max();
  EXPECT_EQ(kMax.InDays(), std::numeric_limits<int>::max());
  static_assert(kMax.InHours() == std::numeric_limits<int>::max());
  static_assert(kMax.InMinutes() == std::numeric_limits<int>::max());
  static_assert(kMax.InSecondsF() == std::numeric_limits<double>::infinity());
  static_assert(kMax.InSeconds() == std::numeric_limits<int64_t>::max());
  EXPECT_EQ(kMax.InMillisecondsF(), std::numeric_limits<double>::infinity());
  EXPECT_EQ(kMax.InMilliseconds(), std::numeric_limits<int64_t>::max());
  EXPECT_EQ(kMax.InMillisecondsRoundedUp(),
            std::numeric_limits<int64_t>::max());

  static_assert(Days(std::numeric_limits<int64_t>::max()).is_max());

  static_assert(Hours(std::numeric_limits<int64_t>::max()).is_max());

  static_assert(Minutes(std::numeric_limits<int64_t>::max()).is_max());

  constexpr int64_t max_int = std::numeric_limits<int64_t>::max();
  constexpr int64_t min_int = std::numeric_limits<int64_t>::min();

  static_assert(Seconds(max_int / Time::kMicrosecondsPerSecond + 1).is_max());

  static_assert(
      Milliseconds(max_int / Time::kMillisecondsPerSecond + 1).is_max());

  static_assert(Microseconds(max_int).is_max());

  static_assert(Seconds(min_int / Time::kMicrosecondsPerSecond - 1).is_min());

  static_assert(
      Milliseconds(min_int / Time::kMillisecondsPerSecond - 1).is_min());

  static_assert(Microseconds(min_int).is_min());

  static_assert(Microseconds(std::numeric_limits<int64_t>::min()).is_min());

  static_assert(Seconds(std::numeric_limits<double>::infinity()).is_max());

  // Note that max_int/min_int will be rounded when converted to doubles - they
  // can't be exactly represented.
  constexpr double max_d = static_cast<double>(max_int);
  constexpr double min_d = static_cast<double>(min_int);

  static_assert(Seconds(max_d / Time::kMicrosecondsPerSecond + 1).is_max());

  static_assert(
      Microseconds(max_d).is_max(),
      "Make sure that 2^63 correctly gets clamped to `max` (crbug.com/612601)");

  static_assert(Milliseconds(std::numeric_limits<double>::infinity()).is_max());

  static_assert(
      Milliseconds(max_d / Time::kMillisecondsPerSecond * 2).is_max());

  static_assert(Seconds(min_d / Time::kMicrosecondsPerSecond - 1).is_min());

  static_assert(
      Milliseconds(min_d / Time::kMillisecondsPerSecond * 2).is_min());
}

TEST(TimeDelta, MinConversions) {
  constexpr TimeDelta kMin = TimeDelta::Min();

  EXPECT_EQ(kMin.InDays(), std::numeric_limits<int>::min());
  static_assert(kMin.InHours() == std::numeric_limits<int>::min());
  static_assert(kMin.InMinutes() == std::numeric_limits<int>::min());
  static_assert(kMin.InSecondsF() == -std::numeric_limits<double>::infinity());
  static_assert(kMin.InSeconds() == std::numeric_limits<int64_t>::min());
  EXPECT_EQ(kMin.InMillisecondsF(), -std::numeric_limits<double>::infinity());
  EXPECT_EQ(kMin.InMilliseconds(), std::numeric_limits<int64_t>::min());
  EXPECT_EQ(kMin.InMillisecondsRoundedUp(),
            std::numeric_limits<int64_t>::min());
}

TEST(TimeDelta, FiniteMaxMin) {
  constexpr TimeDelta kFiniteMax = TimeDelta::FiniteMax();
  constexpr TimeDelta kUnit = Microseconds(1);
  static_assert(kFiniteMax + kUnit == TimeDelta::Max());
  static_assert(kFiniteMax - kUnit < kFiniteMax);

  constexpr TimeDelta kFiniteMin = TimeDelta::FiniteMin();
  static_assert(kFiniteMin - kUnit == TimeDelta::Min());
  static_assert(kFiniteMin + kUnit > kFiniteMin);
}

TEST(TimeDelta, NumericOperators) {
  constexpr double d = 0.5;
  static_assert(Milliseconds(500) == Milliseconds(1000) * d);
  static_assert(Milliseconds(2000) == (Milliseconds(1000) / d));
  static_assert(Milliseconds(500) == (Milliseconds(1000) *= d));
  static_assert(Milliseconds(2000) == (Milliseconds(1000) /= d));
  static_assert(Milliseconds(500) == d * Milliseconds(1000));

  constexpr float f = 0.5;
  static_assert(Milliseconds(500) == Milliseconds(1000) * f);
  static_assert(Milliseconds(2000) == (Milliseconds(1000) / f));
  static_assert(Milliseconds(500) == (Milliseconds(1000) *= f));
  static_assert(Milliseconds(2000) == (Milliseconds(1000) /= f));
  static_assert(Milliseconds(500) == f * Milliseconds(1000));

  constexpr int i = 2;
  static_assert(Milliseconds(2000) == Milliseconds(1000) * i);
  static_assert(Milliseconds(500) == (Milliseconds(1000) / i));
  static_assert(Milliseconds(2000) == (Milliseconds(1000) *= i));
  static_assert(Milliseconds(500) == (Milliseconds(1000) /= i));
  static_assert(Milliseconds(2000) == i * Milliseconds(1000));

  constexpr int64_t i64 = 2;
  static_assert(Milliseconds(2000) == Milliseconds(1000) * i64);
  static_assert(Milliseconds(500) == (Milliseconds(1000) / i64));
  static_assert(Milliseconds(2000) == (Milliseconds(1000) *= i64));
  static_assert(Milliseconds(500) == (Milliseconds(1000) /= i64));
  static_assert(Milliseconds(2000) == i64 * Milliseconds(1000));

  static_assert(Milliseconds(500) == Milliseconds(1000) * 0.5);
  static_assert(Milliseconds(2000) == (Milliseconds(1000) / 0.5));
  static_assert(Milliseconds(500) == (Milliseconds(1000) *= 0.5));
  static_assert(Milliseconds(2000) == (Milliseconds(1000) /= 0.5));
  static_assert(Milliseconds(500) == 0.5 * Milliseconds(1000));

  static_assert(Milliseconds(2000) == Milliseconds(1000) * 2);
  static_assert(Milliseconds(500) == (Milliseconds(1000) / 2));
  static_assert(Milliseconds(2000) == (Milliseconds(1000) *= 2));
  static_assert(Milliseconds(500) == (Milliseconds(1000) /= 2));
  static_assert(Milliseconds(2000) == 2 * Milliseconds(1000));
}

// Basic test of operators between TimeDeltas (without overflow -- next test
// handles overflow).
TEST(TimeDelta, TimeDeltaOperators) {
  constexpr TimeDelta kElevenSeconds = Seconds(11);
  constexpr TimeDelta kThreeSeconds = Seconds(3);

  static_assert(Seconds(14) == kElevenSeconds + kThreeSeconds);
  static_assert(Seconds(14) == kThreeSeconds + kElevenSeconds);
  static_assert(Seconds(8) == kElevenSeconds - kThreeSeconds);
  static_assert(Seconds(-8) == kThreeSeconds - kElevenSeconds);
  static_assert(11.0 / 3.0 == kElevenSeconds / kThreeSeconds);
  static_assert(3.0 / 11.0 == kThreeSeconds / kElevenSeconds);
  static_assert(3 == kElevenSeconds.IntDiv(kThreeSeconds));
  static_assert(0 == kThreeSeconds.IntDiv(kElevenSeconds));
  static_assert(Seconds(2) == kElevenSeconds % kThreeSeconds);
}

TEST(TimeDelta, Overflows) {
  // Some sanity checks. static_asserts used where possible to verify constexpr
  // evaluation at the same time.
  static_assert(TimeDelta::Max().is_max());
  static_assert(-TimeDelta::Max() < TimeDelta());
  static_assert(-TimeDelta::Max() == TimeDelta::Min());
  static_assert(TimeDelta() > -TimeDelta::Max());

  static_assert(TimeDelta::Min().is_min());
  static_assert(-TimeDelta::Min() > TimeDelta());
  static_assert(-TimeDelta::Min() == TimeDelta::Max());
  static_assert(TimeDelta() < -TimeDelta::Min());

  constexpr TimeDelta kLargeDelta = TimeDelta::Max() - Milliseconds(1);
  constexpr TimeDelta kLargeNegative = -kLargeDelta;
  static_assert(TimeDelta() > kLargeNegative);
  static_assert(!kLargeDelta.is_max());
  static_assert(!(-kLargeNegative).is_min());

  // Test +, -, * and / operators.
  constexpr TimeDelta kOneSecond = Seconds(1);
  static_assert((kLargeDelta + kOneSecond).is_max());
  static_assert((kLargeNegative + (-kOneSecond)).is_min());
  static_assert((kLargeNegative - kOneSecond).is_min());
  static_assert((kLargeDelta - (-kOneSecond)).is_max());
  static_assert((kLargeDelta * 2).is_max());
  static_assert((kLargeDelta * -2).is_min());
  static_assert((kLargeDelta / 0.5).is_max());
  static_assert((kLargeDelta / -0.5).is_min());

  // Test math operators on Max() and Min() values
  // Calculations that would overflow are saturated.
  static_assert(TimeDelta::Max() + kOneSecond == TimeDelta::Max());
  static_assert(TimeDelta::Max() * 7 == TimeDelta::Max());
  static_assert(TimeDelta::FiniteMax() + kOneSecond == TimeDelta::Max());
  static_assert(TimeDelta::Min() - kOneSecond == TimeDelta::Min());
  static_assert(TimeDelta::Min() * 7 == TimeDelta::Min());
  static_assert(TimeDelta::FiniteMin() - kOneSecond == TimeDelta::Min());

  // Division is done by converting to double with Max()/Min() converted to
  // +/- infinities.
  static_assert(TimeDelta::Max() / kOneSecond ==
                std::numeric_limits<double>::infinity());
  static_assert(TimeDelta::Max() / -kOneSecond ==
                -std::numeric_limits<double>::infinity());
  static_assert(TimeDelta::Min() / kOneSecond ==
                -std::numeric_limits<double>::infinity());
  static_assert(TimeDelta::Min() / -kOneSecond ==
                std::numeric_limits<double>::infinity());
  static_assert(TimeDelta::Max().IntDiv(kOneSecond) ==
                std::numeric_limits<int64_t>::max());
  static_assert(TimeDelta::Max().IntDiv(-kOneSecond) ==
                std::numeric_limits<int64_t>::min());
  static_assert(TimeDelta::Min().IntDiv(kOneSecond) ==
                std::numeric_limits<int64_t>::min());
  static_assert(TimeDelta::Min().IntDiv(-kOneSecond) ==
                std::numeric_limits<int64_t>::max());
  static_assert(TimeDelta::Max() % kOneSecond == TimeDelta::Max());
  static_assert(TimeDelta::Max() % -kOneSecond == TimeDelta::Max());
  static_assert(TimeDelta::Min() % kOneSecond == TimeDelta::Min());
  static_assert(TimeDelta::Min() % -kOneSecond == TimeDelta::Min());

  // Division by zero.
  static_assert((kOneSecond / 0).is_max());
  static_assert((-kOneSecond / 0).is_min());
  static_assert((TimeDelta::Max() / 0).is_max());
  static_assert((TimeDelta::Min() / 0).is_min());
  EXPECT_EQ(std::numeric_limits<double>::infinity(), kOneSecond / TimeDelta());
  EXPECT_EQ(-std::numeric_limits<double>::infinity(),
            -kOneSecond / TimeDelta());
  EXPECT_EQ(std::numeric_limits<double>::infinity(),
            TimeDelta::Max() / TimeDelta());
  EXPECT_EQ(-std::numeric_limits<double>::infinity(),
            TimeDelta::Min() / TimeDelta());
  static_assert(kOneSecond.IntDiv(TimeDelta()) ==
                std::numeric_limits<int64_t>::max());
  static_assert((-kOneSecond).IntDiv(TimeDelta()) ==
                std::numeric_limits<int64_t>::min());
  static_assert(TimeDelta::Max().IntDiv(TimeDelta()) ==
                std::numeric_limits<int64_t>::max());
  static_assert(TimeDelta::Min().IntDiv(TimeDelta()) ==
                std::numeric_limits<int64_t>::min());
  static_assert(kOneSecond % TimeDelta() == kOneSecond);
  static_assert(-kOneSecond % TimeDelta() == -kOneSecond);
  static_assert(TimeDelta::Max() % TimeDelta() == TimeDelta::Max());
  static_assert(TimeDelta::Min() % TimeDelta() == TimeDelta::Min());

  // Division by infinity.
  static_assert(kLargeDelta / TimeDelta::Min() == 0);
  static_assert(kLargeDelta / TimeDelta::Max() == 0);
  static_assert(kLargeNegative / TimeDelta::Min() == 0);
  static_assert(kLargeNegative / TimeDelta::Max() == 0);
  static_assert(kLargeDelta.IntDiv(TimeDelta::Min()) == 0);
  static_assert(kLargeDelta.IntDiv(TimeDelta::Max()) == 0);
  static_assert(kLargeNegative.IntDiv(TimeDelta::Min()) == 0);
  static_assert(kLargeNegative.IntDiv(TimeDelta::Max()) == 0);
  static_assert(kOneSecond % TimeDelta::Min() == kOneSecond);
  static_assert(kOneSecond % TimeDelta::Max() == kOneSecond);

  // Test that double conversions overflow to infinity.
  static_assert((kLargeDelta + kOneSecond).InSecondsF() ==
                std::numeric_limits<double>::infinity());
  EXPECT_EQ((kLargeDelta + kOneSecond).InMillisecondsF(),
            std::numeric_limits<double>::infinity());
  EXPECT_EQ((kLargeDelta + kOneSecond).InMicrosecondsF(),
            std::numeric_limits<double>::infinity());

  // Test op=.
  static_assert((TimeDelta::FiniteMax() += kOneSecond).is_max());
  static_assert((TimeDelta::FiniteMin() += -kOneSecond).is_min());

  static_assert((TimeDelta::FiniteMin() -= kOneSecond).is_min());
  static_assert((TimeDelta::FiniteMax() -= -kOneSecond).is_max());

  static_assert((TimeDelta::FiniteMax() *= 2).is_max());
  static_assert((TimeDelta::FiniteMin() *= 1.5).is_min());

  static_assert((TimeDelta::FiniteMax() /= 0.5).is_max());
  static_assert((TimeDelta::FiniteMin() /= 0.5).is_min());

  static_assert((Seconds(1) %= TimeDelta::Max()) == Seconds(1));
  static_assert((Seconds(1) %= TimeDelta()) == Seconds(1));

  // Test operations with Time and TimeTicks.
  EXPECT_TRUE((kLargeDelta + Time::Now()).is_max());
  EXPECT_TRUE((kLargeDelta + TimeTicks::Now()).is_max());
  EXPECT_TRUE((Time::Now() + kLargeDelta).is_max());
  EXPECT_TRUE((TimeTicks::Now() + kLargeDelta).is_max());

  Time time_now = Time::Now();
  EXPECT_EQ(kOneSecond, (time_now + kOneSecond) - time_now);
  EXPECT_EQ(-kOneSecond, (time_now - kOneSecond) - time_now);

  TimeTicks ticks_now = TimeTicks::Now();
  EXPECT_EQ(-kOneSecond, (ticks_now - kOneSecond) - ticks_now);
  EXPECT_EQ(kOneSecond, (ticks_now + kOneSecond) - ticks_now);
}

TEST(TimeDelta, CeilToMultiple) {
  for (const auto interval : {Seconds(10), Seconds(-10)}) {
    SCOPED_TRACE(interval);
    EXPECT_EQ(TimeDelta().CeilToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(1).CeilToMultiple(interval), Seconds(10));
    EXPECT_EQ(Seconds(9).CeilToMultiple(interval), Seconds(10));
    EXPECT_EQ(Seconds(10).CeilToMultiple(interval), Seconds(10));
    EXPECT_EQ(Seconds(15).CeilToMultiple(interval), Seconds(20));
    EXPECT_EQ(Seconds(20).CeilToMultiple(interval), Seconds(20));
    EXPECT_EQ(TimeDelta::Max().CeilToMultiple(interval), TimeDelta::Max());
    EXPECT_EQ(Seconds(-1).CeilToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(-9).CeilToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(-10).CeilToMultiple(interval), Seconds(-10));
    EXPECT_EQ(Seconds(-15).CeilToMultiple(interval), Seconds(-10));
    EXPECT_EQ(Seconds(-20).CeilToMultiple(interval), Seconds(-20));
    EXPECT_EQ(TimeDelta::Min().CeilToMultiple(interval), TimeDelta::Min());
  }

  for (const auto interval : {TimeDelta::Max(), TimeDelta::Min()}) {
    SCOPED_TRACE(interval);
    EXPECT_EQ(TimeDelta().CeilToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(1).CeilToMultiple(interval), TimeDelta::Max());
    EXPECT_EQ(Seconds(9).CeilToMultiple(interval), TimeDelta::Max());
    EXPECT_EQ(Seconds(10).CeilToMultiple(interval), TimeDelta::Max());
    EXPECT_EQ(Seconds(15).CeilToMultiple(interval), TimeDelta::Max());
    EXPECT_EQ(Seconds(20).CeilToMultiple(interval), TimeDelta::Max());
    EXPECT_EQ(TimeDelta::Max().CeilToMultiple(interval), TimeDelta::Max());
    EXPECT_EQ(Seconds(-1).CeilToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(-9).CeilToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(-10).CeilToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(-15).CeilToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(-20).CeilToMultiple(interval), TimeDelta());
    EXPECT_EQ(TimeDelta::Min().CeilToMultiple(interval), TimeDelta::Min());
  }
}

TEST(TimeDelta, FloorToMultiple) {
  for (const auto interval : {Seconds(10), Seconds(-10)}) {
    SCOPED_TRACE(interval);
    EXPECT_EQ(TimeDelta().FloorToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(1).FloorToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(9).FloorToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(10).FloorToMultiple(interval), Seconds(10));
    EXPECT_EQ(Seconds(15).FloorToMultiple(interval), Seconds(10));
    EXPECT_EQ(Seconds(20).FloorToMultiple(interval), Seconds(20));
    EXPECT_EQ(TimeDelta::Max().FloorToMultiple(interval), TimeDelta::Max());
    EXPECT_EQ(Seconds(-1).FloorToMultiple(interval), Seconds(-10));
    EXPECT_EQ(Seconds(-9).FloorToMultiple(interval), Seconds(-10));
    EXPECT_EQ(Seconds(-10).FloorToMultiple(interval), Seconds(-10));
    EXPECT_EQ(Seconds(-15).FloorToMultiple(interval), Seconds(-20));
    EXPECT_EQ(Seconds(-20).FloorToMultiple(interval), Seconds(-20));
    EXPECT_EQ(TimeDelta::Min().FloorToMultiple(interval), TimeDelta::Min());
  }

  for (const auto interval : {TimeDelta::Max(), TimeDelta::Min()}) {
    SCOPED_TRACE(interval);
    EXPECT_EQ(TimeDelta().FloorToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(1).FloorToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(9).FloorToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(10).FloorToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(15).FloorToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(20).FloorToMultiple(interval), TimeDelta());
    EXPECT_EQ(TimeDelta::Max().FloorToMultiple(interval), TimeDelta::Max());
    EXPECT_EQ(Seconds(-1).FloorToMultiple(interval), TimeDelta::Min());
    EXPECT_EQ(Seconds(-9).FloorToMultiple(interval), TimeDelta::Min());
    EXPECT_EQ(Seconds(-10).FloorToMultiple(interval), TimeDelta::Min());
    EXPECT_EQ(Seconds(-15).FloorToMultiple(interval), TimeDelta::Min());
    EXPECT_EQ(Seconds(-20).FloorToMultiple(interval), TimeDelta::Min());
    EXPECT_EQ(TimeDelta::Min().FloorToMultiple(interval), TimeDelta::Min());
  }
}

TEST(TimeDelta, RoundToMultiple) {
  for (const auto interval : {Seconds(10), Seconds(-10)}) {
    SCOPED_TRACE(interval);
    EXPECT_EQ(TimeDelta().RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(1).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(9).RoundToMultiple(interval), Seconds(10));
    EXPECT_EQ(Seconds(10).RoundToMultiple(interval), Seconds(10));
    EXPECT_EQ(Seconds(15).RoundToMultiple(interval), Seconds(20));
    EXPECT_EQ(Seconds(20).RoundToMultiple(interval), Seconds(20));
    EXPECT_EQ(TimeDelta::Max().RoundToMultiple(interval), TimeDelta::Max());
    EXPECT_EQ(Seconds(-1).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(-9).RoundToMultiple(interval), Seconds(-10));
    EXPECT_EQ(Seconds(-10).RoundToMultiple(interval), Seconds(-10));
    EXPECT_EQ(Seconds(-15).RoundToMultiple(interval), Seconds(-20));
    EXPECT_EQ(Seconds(-20).RoundToMultiple(interval), Seconds(-20));
    EXPECT_EQ(TimeDelta::Min().RoundToMultiple(interval), TimeDelta::Min());
  }

  for (const auto interval : {TimeDelta::Max(), TimeDelta::Min()}) {
    SCOPED_TRACE(interval);
    EXPECT_EQ(TimeDelta().RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(1).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(9).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(10).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(15).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(20).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(TimeDelta::Max().RoundToMultiple(interval), TimeDelta::Max());
    EXPECT_EQ(Seconds(-1).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(-9).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(-10).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(-15).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(Seconds(-20).RoundToMultiple(interval), TimeDelta());
    EXPECT_EQ(TimeDelta::Min().RoundToMultiple(interval), TimeDelta::Min());
  }
}

TEST(TimeBase, AddSubDeltaSaturates) {
  constexpr TimeTicks kLargeTimeTicks =
      TimeTicks() + Microseconds(std::numeric_limits<int64_t>::max() - 1);

  constexpr TimeTicks kLargeNegativeTimeTicks =
      TimeTicks() + Microseconds(std::numeric_limits<int64_t>::min() + 1);

  static_assert((kLargeTimeTicks + TimeDelta::Max()).is_max());
  static_assert((kLargeNegativeTimeTicks + TimeDelta::Max()).is_max());
  static_assert((kLargeTimeTicks - TimeDelta::Max()).is_min());
  static_assert((kLargeNegativeTimeTicks - TimeDelta::Max()).is_min());
  static_assert((TimeTicks() + TimeDelta::Max()).is_max());
  static_assert((TimeTicks() - TimeDelta::Max()).is_min());
  EXPECT_TRUE((TimeTicks::Now() + TimeDelta::Max()).is_max())
      << (TimeTicks::Now() + TimeDelta::Max());
  EXPECT_TRUE((TimeTicks::Now() - TimeDelta::Max()).is_min())
      << (TimeTicks::Now() - TimeDelta::Max());

  static_assert((kLargeTimeTicks + TimeDelta::Min()).is_min());
  static_assert((kLargeNegativeTimeTicks + TimeDelta::Min()).is_min());
  static_assert((kLargeTimeTicks - TimeDelta::Min()).is_max());
  static_assert((kLargeNegativeTimeTicks - TimeDelta::Min()).is_max());
  static_assert((TimeTicks() + TimeDelta::Min()).is_min());
  static_assert((TimeTicks() - TimeDelta::Min()).is_max());
  EXPECT_TRUE((TimeTicks::Now() + TimeDelta::Min()).is_min())
      << (TimeTicks::Now() + TimeDelta::Min());
  EXPECT_TRUE((TimeTicks::Now() - TimeDelta::Min()).is_max())
      << (TimeTicks::Now() - TimeDelta::Min());
}

TEST(TimeBase, AddSubInfinities) {
  /*
  TODO(chokobole):
  // CHECK when adding opposite signs or subtracting same sign.
  EXPECT_CHECK_DEATH({ TimeTicks::Min() + TimeDelta::Max(); });
  EXPECT_CHECK_DEATH({ TimeTicks::Max() + TimeDelta::Min(); });
  EXPECT_CHECK_DEATH({ TimeTicks::Min() - TimeDelta::Min(); });
  EXPECT_CHECK_DEATH({ TimeTicks::Max() - TimeDelta::Max(); });
  */

  // Saturates when adding same sign or subtracting opposite signs.
  static_assert((TimeTicks::Max() + TimeDelta::Max()).is_max());
  static_assert((TimeTicks::Min() + TimeDelta::Min()).is_min());
  static_assert((TimeTicks::Max() - TimeDelta::Min()).is_max());
  static_assert((TimeTicks::Min() - TimeDelta::Max()).is_min());
}

constexpr TimeDelta TestTimeDeltaConstexprCopyAssignment() {
  TimeDelta a = Seconds(1);
  TimeDelta b;
  b = a;
  return b;
}

TEST(TimeDelta, ConstexprAndTriviallyCopiable) {
  // "Trivially copyable" is necessary for use in std::atomic<TimeDelta>.
  static_assert(std::is_trivially_copyable<TimeDelta>());

  // Copy ctor.
  constexpr TimeDelta a = Seconds(1);
  constexpr TimeDelta b{a};
  static_assert(a == b);

  // Copy assignment.
  static_assert(a == TestTimeDeltaConstexprCopyAssignment());
}

TEST(TimeDeltaLogging, DCheckEqCompiles) {
  DCHECK_EQ(TimeDelta(), TimeDelta());
}

TEST(TimeDeltaLogging, EmptyIsZero) {
  constexpr TimeDelta kZero;
  EXPECT_EQ("0 s", AnyToString(kZero));
}

TEST(TimeDeltaLogging, FiveHundredMs) {
  constexpr TimeDelta kFiveHundredMs = Milliseconds(500);
  EXPECT_EQ("0.5 s", AnyToString(kFiveHundredMs));
}

TEST(TimeDeltaLogging, MinusTenSeconds) {
  constexpr TimeDelta kMinusTenSeconds = Seconds(-10);
  EXPECT_EQ("-10 s", AnyToString(kMinusTenSeconds));
}

TEST(TimeDeltaLogging, DoesNotMessUpFormattingFlags) {
  std::ostringstream oss;
  std::ios_base::fmtflags flags_before = oss.flags();
  oss << TimeDelta();
  EXPECT_EQ(flags_before, oss.flags());
}

TEST(TimeDeltaLogging, DoesNotMakeStreamBad) {
  std::ostringstream oss;
  oss << TimeDelta();
  EXPECT_TRUE(oss.good());
}

TEST(TimeLogging, DCheckEqCompiles) { DCHECK_EQ(Time(), Time()); }

TEST(TimeLogging, DoesNotMessUpFormattingFlags) {
  std::ostringstream oss;
  std::ios_base::fmtflags flags_before = oss.flags();
  oss << Time();
  EXPECT_EQ(flags_before, oss.flags());
}

TEST(TimeLogging, DoesNotMakeStreamBad) {
  std::ostringstream oss;
  oss << Time();
  EXPECT_TRUE(oss.good());
}

TEST(TimeTicksLogging, DCheckEqCompiles) {
  DCHECK_EQ(TimeTicks(), TimeTicks());
}

TEST(TimeTicksLogging, ZeroTime) {
  TimeTicks zero;
  EXPECT_EQ("0 bogo-microseconds", AnyToString(zero));
}

TEST(TimeTicksLogging, FortyYearsLater) {
  TimeTicks forty_years_later = TimeTicks() + Days(365.25 * 40);
  EXPECT_EQ("1262304000000000 bogo-microseconds",
            AnyToString(forty_years_later));
}

TEST(TimeTicksLogging, DoesNotMessUpFormattingFlags) {
  std::ostringstream oss;
  std::ios_base::fmtflags flags_before = oss.flags();
  oss << TimeTicks();
  EXPECT_EQ(flags_before, oss.flags());
}

TEST(TimeTicksLogging, DoesNotMakeStreamBad) {
  std::ostringstream oss;
  oss << TimeTicks();
  EXPECT_TRUE(oss.good());
}

TEST(TimeDelta, ChronoConversion) {
  TimeDelta delta = Seconds(1);
  std::chrono::microseconds microseconds = delta.ToChronoMicroseconds();
  EXPECT_EQ(microseconds.count(), delta.InMicroseconds());
  EXPECT_EQ(TimeDelta::FromChrono(microseconds), delta);

  delta = Hours(1);
  std::chrono::minutes minutes = delta.ToChronoMinutes();
  EXPECT_EQ(minutes.count(), delta.InMinutes());
  EXPECT_EQ(TimeDelta::FromChrono(minutes), delta);
}

// Test conversion to/from TimeDeltas elapsed since the Windows epoch.
// Conversions should be idempotent and non-lossy.
TEST(Time, DeltaSinceUnixEpoch) {
  const TimeDelta delta = Microseconds(123);
  EXPECT_EQ(delta,
            Time::FromDeltaSinceUnixEpoch(delta).ToDeltaSinceUnixEpoch());

  const Time now = Time::Now();
  const Time actual =
      Time::FromDeltaSinceUnixEpoch(now.ToDeltaSinceUnixEpoch());
  EXPECT_EQ(now, actual);

  // Null times should remain null after a round-trip conversion. This is an
  // important invariant for the common use case of serialization +
  // deserialization.
  const Time should_be_null =
      Time::FromDeltaSinceUnixEpoch(Time().ToDeltaSinceUnixEpoch());
  EXPECT_TRUE(should_be_null.is_null());
}

// Test conversion to/from time_t.
TEST(TimeTest, TimeT) {
  EXPECT_EQ(10, Time().FromTimeT(10).ToTimeT());
  EXPECT_EQ(10.0, Time().FromTimeT(10).ToDoubleT());

  // Conversions of 0 should stay 0.
  EXPECT_EQ(0, Time().ToTimeT());
}

#if BUILDFLAG(IS_POSIX)
TEST(Time, FromTimeVal) {
  Time now = Time::Now();
  Time also_now = Time::FromTimeVal(now.ToTimeVal());
  EXPECT_EQ(now, also_now);
}
#endif  // BUILDFLAG(IS_POSIX)

TEST(Time, ChronoConversion) {
  Time now = Time::Now();
  std::chrono::system_clock::time_point tp = now.ToChrono();
  EXPECT_EQ(tp.time_since_epoch(),
            now.ToDeltaSinceUnixEpoch()
                .ToChrono<std::chrono::system_clock::duration>());
  EXPECT_NEAR((now - Time()).InMicroseconds(),
              (Time::FromChrono(tp) - Time()).InMicroseconds(), 1);
}

TEST(TimeTicks, ChronoConversion) {
  TimeTicks now = TimeTicks::Now();
  std::chrono::steady_clock::time_point tp = now.ToChrono();
  EXPECT_EQ(
      tp.time_since_epoch(),
      (now - TimeTicks{}).ToChrono<std::chrono::steady_clock::duration>());
  EXPECT_NEAR((now - TimeTicks()).InMicroseconds(),
              (TimeTicks::FromChrono(tp) - TimeTicks()).InMicroseconds(), 1);
}

}  // namespace base
}  // namespace tachyon
