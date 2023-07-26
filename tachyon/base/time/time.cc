// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/time/time.h"

#include "absl/time/clock.h"

namespace tachyon {
namespace base {

// TimeDelta ------------------------------------------------------------------

// static
TimeDelta TimeDelta::FromAbsl(absl::Duration d) {
  return TimeDelta(absl::ToInt64Microseconds(d));
}

std::chrono::hours TimeDelta::ToChronoHours() const {
  return std::chrono::hours(InHours());
}

std::chrono::minutes TimeDelta::ToChronoMinutes() const {
  return std::chrono::minutes(InMinutes());
}

std::chrono::seconds TimeDelta::ToChronoSeconds() const {
  return std::chrono::seconds(InSeconds());
}

std::chrono::milliseconds TimeDelta::ToChronoMilliseconds() const {
  return std::chrono::milliseconds(InMilliseconds());
}

std::chrono::nanoseconds TimeDelta::ToChronoNanoseconds() const {
  return std::chrono::nanoseconds(InNanoseconds());
}

int64_t TimeDelta::InSecondsFloored() const {
  if (!is_inf()) {
    const int64_t result = delta_ / Time::kMicrosecondsPerSecond;
    // Convert |result| from truncating to flooring.
    return (result * Time::kMicrosecondsPerSecond > delta_) ? (result - 1)
                                                            : result;
  }
  return delta_;
}

int TimeDelta::InDays() const {
  if (!is_inf()) return static_cast<int>(delta_ / Time::kMicrosecondsPerDay);
  return (delta_ < 0) ? std::numeric_limits<int>::min()
                      : std::numeric_limits<int>::max();
}

int TimeDelta::InDaysFloored() const {
  if (!is_inf()) {
    const int result = delta_ / Time::kMicrosecondsPerDay;
    // Convert |result| from truncating to flooring.
    return (result * Time::kMicrosecondsPerDay > delta_) ? (result - 1)
                                                         : result;
  }
  return (delta_ < 0) ? std::numeric_limits<int>::min()
                      : std::numeric_limits<int>::max();
}

double TimeDelta::InMillisecondsF() const {
  if (!is_inf())
    return static_cast<double>(delta_) / Time::kMicrosecondsPerMillisecond;
  return (delta_ < 0) ? -std::numeric_limits<double>::infinity()
                      : std::numeric_limits<double>::infinity();
}

int64_t TimeDelta::InMilliseconds() const {
  if (!is_inf()) return delta_ / Time::kMicrosecondsPerMillisecond;
  return (delta_ < 0) ? std::numeric_limits<int64_t>::min()
                      : std::numeric_limits<int64_t>::max();
}

int64_t TimeDelta::InMillisecondsRoundedUp() const {
  if (!is_inf()) {
    const int64_t result = delta_ / Time::kMicrosecondsPerMillisecond;
    // Convert |result| from truncating to ceiling.
    return (delta_ > result * Time::kMicrosecondsPerMillisecond) ? (result + 1)
                                                                 : result;
  }
  return delta_;
}

double TimeDelta::InMicrosecondsF() const {
  if (!is_inf()) return static_cast<double>(delta_);
  return (delta_ < 0) ? -std::numeric_limits<double>::infinity()
                      : std::numeric_limits<double>::infinity();
}

TimeDelta TimeDelta::CeilToMultiple(TimeDelta interval) const {
  if (is_inf() || interval.is_zero()) return *this;
  const TimeDelta remainder = *this % interval;
  if (delta_ < 0) return *this - remainder;
  return remainder.is_zero() ? *this
                             : (*this - remainder + interval.magnitude());
}

TimeDelta TimeDelta::FloorToMultiple(TimeDelta interval) const {
  if (is_inf() || interval.is_zero()) return *this;
  const TimeDelta remainder = *this % interval;
  if (delta_ < 0) {
    return remainder.is_zero() ? *this
                               : (*this - remainder - interval.magnitude());
  }
  return *this - remainder;
}

TimeDelta TimeDelta::RoundToMultiple(TimeDelta interval) const {
  if (is_inf() || interval.is_zero()) return *this;
  if (interval.is_inf()) return TimeDelta();
  const TimeDelta half = interval.magnitude() / 2;
  return (delta_ < 0) ? (*this - half).CeilToMultiple(interval)
                      : (*this + half).FloorToMultiple(interval);
}

std::ostream& operator<<(std::ostream& os, TimeDelta time_delta) {
  return os << time_delta.InSecondsF() << " s";
}

// Time -----------------------------------------------------------------------

// static
Time Time::Now() {
  absl::Time now = absl::Now();
  return Time(absl::ToUnixMicros(now));
}

time_t Time::ToTimeT() const {
  if (is_null()) return 0;  // Preserve 0 so we can tell it doesn't exist.
  if (!is_inf()) {
    return saturated_cast<time_t>((*this - Time()).InSecondsFloored());
  }
  return (us_ < 0) ? std::numeric_limits<time_t>::min()
                   : std::numeric_limits<time_t>::max();
}

// static
Time Time::FromDoubleT(double dt) {
  // Preserve 0 so we can tell it doesn't exist.
  return (dt == 0 || std::isnan(dt)) ? Time() : (Time() + Seconds(dt));
}

double Time::ToDoubleT() const {
  if (is_null()) return 0;  // Preserve 0 so we can tell it doesn't exist.
  if (!is_inf()) return (*this - Time()).InSecondsF();
  return (us_ < 0) ? -std::numeric_limits<double>::infinity()
                   : std::numeric_limits<double>::infinity();
}

#if BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
// static
Time Time::FromTimeSpec(const timespec& ts) {
  return FromDoubleT(ts.tv_sec +
                     static_cast<double>(ts.tv_nsec) / kNanosecondsPerSecond);
}
#endif

// static
Time Time::FromAbsl(absl::Time time) { return Time(absl::ToUnixMicros(time)); }

absl::Time Time::ToAbsl() const { return absl::FromUnixMicros(us_); }

// static
Time Time::FromChrono(std::chrono::system_clock::time_point tp) {
  return Time::FromDeltaSinceUnixEpoch(
      TimeDelta::FromChrono(tp.time_since_epoch()));
}

std::chrono::system_clock::time_point Time::ToChrono() const {
  using D = std::chrono::system_clock::duration;
  return std::chrono::system_clock::time_point{} +
         ToDeltaSinceUnixEpoch().ToChrono<D>();
}

std::ostream& operator<<(std::ostream& os, Time time) {
  const TimeDelta time_delta = time - Time();
  absl::Time absl_time = absl::FromUnixMicros(time_delta.InMicroseconds());
  return os << absl_time;
}

// static
TimeTicks TimeTicks::FromChrono(std::chrono::steady_clock::time_point tp) {
  return TimeTicks(
      TimeDelta::FromChrono(tp.time_since_epoch()).InMicroseconds());
}

std::chrono::steady_clock::time_point TimeTicks::ToChrono() const {
  using D = std::chrono::steady_clock::duration;
  return std::chrono::steady_clock::time_point{} +
         Microseconds(us_).ToChrono<D>();
}

std::ostream& operator<<(std::ostream& os, TimeTicks time_ticks) {
  // This function formats a TimeTicks object as "bogo-microseconds".
  // The origin and granularity of the count are platform-specific, and may very
  // from run to run. Although bogo-microseconds usually roughly correspond to
  // real microseconds, the only real guarantee is that the number never goes
  // down during a single run.
  const TimeDelta as_time_delta = time_ticks - TimeTicks();
  return os << as_time_delta.InMicroseconds() << " bogo-microseconds";
}

// ThreadTicks ----------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, ThreadTicks thread_ticks) {
  const TimeDelta as_time_delta = thread_ticks - ThreadTicks();
  return os << as_time_delta.InMicroseconds() << " bogo-thread-microseconds";
}

}  // namespace base
}  // namespace tachyon
