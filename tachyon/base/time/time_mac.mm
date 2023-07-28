// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <stddef.h>
#include <stdint.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>

#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>

#include "tachyon/base/logging.h"
#include "tachyon/base/mac/mach_logging.h"
#include "tachyon/base/mac/scoped_cftyperef.h"
#include "tachyon/base/mac/scoped_mach_port.h"
#include "tachyon/base/numerics/safe_conversions.h"
#include "tachyon/base/time/time.h"
// #include "tachyon/base/time/time_override.h"
#include "tachyon/build/build_config.h"

#if !BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)
#include <errno.h>
#include <time.h>

#include "tachyon/base/ios/ios_util.h"
#endif  // !BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

namespace {

#if BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)
// Returns a pointer to the initialized Mach timebase info struct.
mach_timebase_info_data_t* MachTimebaseInfo() {
  static mach_timebase_info_data_t timebase_info = []() {
    mach_timebase_info_data_t info;
    kern_return_t kr = mach_timebase_info(&info);
    MACH_DCHECK(kr == KERN_SUCCESS, kr) << "mach_timebase_info";
    DCHECK(info.numer);
    DCHECK(info.denom);
    return info;
  }();
  return &timebase_info;
}

int64_t MachTimeToMicroseconds(uint64_t mach_time) {
  // timebase_info gives us the conversion factor between absolute time tick
  // units and nanoseconds.
  mach_timebase_info_data_t* timebase_info = MachTimebaseInfo();

  // Take the fast path when the conversion is 1:1. The result will for sure fit
  // into an int_64 because we're going from nanoseconds to microseconds.
  if (timebase_info->numer == timebase_info->denom) {
    return static_cast<int64_t>(mach_time / tachyon::base::Time::kNanosecondsPerMicrosecond);
  }

  uint64_t microseconds = 0;
  const uint64_t divisor = timebase_info->denom * tachyon::base::Time::kNanosecondsPerMicrosecond;

  // Microseconds is mach_time * timebase.numer /
  // (timebase.denom * kNanosecondsPerMicrosecond). Divide first to reduce
  // the chance of overflow. Also stash the remainder right now, a likely
  // byproduct of the division.
  microseconds = mach_time / divisor;
  const uint64_t mach_time_remainder = mach_time % divisor;

  // Now multiply, keeping an eye out for overflow.
  CHECK(!__builtin_umulll_overflow(microseconds, timebase_info->numer, &microseconds));

  // By dividing first we lose precision. Regain it by adding back the
  // microseconds from the remainder, with an eye out for overflow.
  uint64_t least_significant_microseconds = (mach_time_remainder * timebase_info->numer) / divisor;
  CHECK(!__builtin_uaddll_overflow(microseconds, least_significant_microseconds, &microseconds));

  // Don't bother with the rollover handling that the Windows version does.
  // The returned time in microseconds is enough for 292,277 years (starting
  // from 2^63 because the returned int64_t is signed,
  // 9223372036854775807 / (1e6 * 60 * 60 * 24 * 365.2425) = 292,277).
  return tachyon::base::checked_cast<int64_t>(microseconds);
}
#endif  // BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)

// Returns monotonically growing number of ticks in microseconds since some
// unspecified starting point.
int64_t ComputeCurrentTicks() {
#if !BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)
  struct timespec tp;
  // clock_gettime() returns 0 on success and -1 on failure. Failure can only
  // happen because of bad arguments (unsupported clock type or timespec pointer
  // out of accessible address space). Here it is known that neither can happen
  // since the timespec parameter is stack allocated right above and
  // `CLOCK_MONOTONIC` is supported on all versions of iOS that Chrome is
  // supported on.
  int res = clock_gettime(CLOCK_MONOTONIC, &tp);
  DCHECK_EQ(res, 0) << "Failed clock_gettime, errno: " << errno;

  return (int64_t)tp.tv_sec * 1000000 + tp.tv_nsec / 1000;
#else
  // mach_absolute_time is it when it comes to ticks on the Mac.  Other calls
  // with less precision (such as TickCount) just call through to
  // mach_absolute_time.
  return MachTimeToMicroseconds(mach_absolute_time());
#endif  // !BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)
}

int64_t ComputeThreadTicks() {
  // The pthreads library keeps a cached reference to the thread port, which
  // does not have to be released like mach_thread_self() does.
  mach_port_t thread_port = pthread_mach_thread_np(pthread_self());
  if (thread_port == MACH_PORT_NULL) {
    DLOG(ERROR) << "Failed to get pthread_mach_thread_np()";
    return 0;
  }

  mach_msg_type_number_t thread_info_count = THREAD_BASIC_INFO_COUNT;
  thread_basic_info_data_t thread_info_data;

  kern_return_t kr =
      thread_info(thread_port, THREAD_BASIC_INFO,
                  reinterpret_cast<thread_info_t>(&thread_info_data), &thread_info_count);
  MACH_DCHECK(kr == KERN_SUCCESS, kr) << "thread_info";

  tachyon::base::CheckedNumeric<int64_t> absolute_micros(thread_info_data.user_time.seconds +
                                                         thread_info_data.system_time.seconds);
  absolute_micros *= tachyon::base::Time::kMicrosecondsPerSecond;
  absolute_micros +=
      (thread_info_data.user_time.microseconds + thread_info_data.system_time.microseconds);
  return absolute_micros.ValueOrDie();
}

}  // namespace

namespace tachyon::base {

// The Time routines in this file use Mach and CoreFoundation APIs, since the
// POSIX definition of time_t in macOS wraps around after 2038--and
// there are already cookie expiration dates, etc., past that time out in
// the field.  Using CFDate prevents that problem, and using mach_absolute_time
// for TimeTicks gives us nice high-resolution interval timing.

// Time -----------------------------------------------------------------------

namespace subtle {
Time TimeNowIgnoringOverride() { return Time::FromCFAbsoluteTime(CFAbsoluteTimeGetCurrent()); }

Time TimeNowFromSystemTimeIgnoringOverride() {
  // Just use TimeNowIgnoringOverride() because it returns the system time.
  return TimeNowIgnoringOverride();
}
}  // namespace subtle

// static
Time Time::FromCFAbsoluteTime(CFAbsoluteTime t) {
  static_assert(std::numeric_limits<CFAbsoluteTime>::has_infinity,
                "CFAbsoluteTime must have an infinity value");
  if (t == 0) return Time();  // Consider 0 as a null Time.
  return (t == std::numeric_limits<CFAbsoluteTime>::infinity())
             ? Max()
             : (Time() + Seconds(double{t + kCFAbsoluteTimeIntervalSince1970}));
}

CFAbsoluteTime Time::ToCFAbsoluteTime() const {
  static_assert(std::numeric_limits<CFAbsoluteTime>::has_infinity,
                "CFAbsoluteTime must have an infinity value");
  if (is_null()) return 0;  // Consider 0 as a null Time.
  return is_max()
             ? std::numeric_limits<CFAbsoluteTime>::infinity()
             : (CFAbsoluteTime{(*this - Time()).InSecondsF()} - kCFAbsoluteTimeIntervalSince1970);
}

// static
Time Time::FromNSDate(NSDate* date) {
  DCHECK(date);
  return FromCFAbsoluteTime(date.timeIntervalSinceReferenceDate);
}

NSDate* Time::ToNSDate() const {
  return [NSDate dateWithTimeIntervalSinceReferenceDate:ToCFAbsoluteTime()];
}

// TimeDelta ------------------------------------------------------------------

#if BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)
// static
TimeDelta TimeDelta::FromMachTime(uint64_t mach_time) {
  return Microseconds(MachTimeToMicroseconds(mach_time));
}
#endif  // BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)

// TimeTicks ------------------------------------------------------------------

// static
TimeTicks TimeTicks::Now() { return TimeTicks() + Microseconds(ComputeCurrentTicks()); }

// static
bool TimeTicks::IsHighResolution() { return true; }

// static
bool TimeTicks::IsConsistentAcrossProcesses() { return true; }

#if BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)
// static
TimeTicks TimeTicks::FromMachAbsoluteTime(uint64_t mach_absolute_time) {
  return TimeTicks(MachTimeToMicroseconds(mach_absolute_time));
}

// static
mach_timebase_info_data_t TimeTicks::SetMachTimebaseInfoForTesting(
    mach_timebase_info_data_t timebase) {
  mach_timebase_info_data_t orig_timebase = *MachTimebaseInfo();

  *MachTimebaseInfo() = timebase;

  return orig_timebase;
}

#endif  // BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)

// ThreadTicks ----------------------------------------------------------------

// static
ThreadTicks ThreadTicks::Now() { return ThreadTicks() + Microseconds(ComputeThreadTicks()); }

}  // namespace tachyon::base
