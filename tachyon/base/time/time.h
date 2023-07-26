// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Followings are modified from chromium/base/time/time.h
// * Unix epoch is being used to represent Time class instead of Windows epoch.
// * Time is implemented using absl::Time inside.

// `Time` represents an absolute point in coordinated universal time (UTC),
// internally represented as microseconds (s/1,000,000) since the Unix epoch
// (1970-01-01 00:00:00 UTC). System-dependent clock interface routines are
// defined in time_PLATFORM.cc. Note that values for `Time` may skew and jump
// around as the operating system makes adjustments to synchronize (e.g., with
// NTP servers). Thus, client code that uses the `Time` class must account for
// this.
//
// `TimeDelta` represents a duration of time, internally represented in
// microseconds.
//
// `TimeTicks` and `ThreadTicks` represent an abstract time that is most of the
// time incrementing, for use in measuring time durations. Internally, they are
// represented in microseconds. They cannot be converted to a human-readable
// time, but are guaranteed not to decrease (unlike the `Time` class). Note
// that `TimeTicks` may "stand still" (e.g., if the computer is suspended), and
// `ThreadTicks` will "stand still" whenever the thread has been de-scheduled
// by the operating system.
//
// All time classes are copyable, assignable, and occupy 64 bits per instance.
// Prefer to pass them by value, e.g.:
//
//   void MyFunction(TimeDelta arg);
//
// All time classes support `operator<<` with logging streams, e.g. `LOG(INFO)`.
//
// Example use cases for different time classes:
//
//   Time:        Interpreting the wall-clock time provided by a remote system.
//                Detecting whether cached resources have expired. Providing the
//                user with a display of the current date and time. Determining
//                the amount of time between events across re-boots of the
//                machine.
//
//   TimeTicks:   Tracking the amount of time a task runs. Executing delayed
//                tasks at the right time. Computing presentation timestamps.
//                Synchronizing audio and video using TimeTicks as a common
//                reference clock (lip-sync). Measuring network round-trip
//                latency.
//
//   ThreadTicks: Benchmarking how long the current thread has been doing actual
//                work.
//
// Serialization:
//
// - Time: use `FromDeltaSinceUnixEpoch()`/`ToDeltaSinceUnixEpoch()`.
// - TimeDelta: use `Microseconds()`/`InMicroseconds()`.
//
// `TimeTicks` and `ThreadTicks` do not have a stable origin; serialization for
// the purpose of persistence is not supported.

#ifndef TACHYON_BASE_TIME_TIME_H_
#define TACHYON_BASE_TIME_TIME_H_

#include <stdint.h>
#include <time.h>

#include <chrono>
#include <iosfwd>
#include <limits>
#include <ostream>

#include "absl/time/time.h"
#include "gtest/gtest_prod.h"

#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/numerics/clamped_math.h"
#include "tachyon/build/build_config.h"
#include "tachyon/export.h"

#if BUILDFLAG(IS_APPLE)
#include "tachyon/base/time/buildflags/buildflags.h"
#endif

#if BUILDFLAG(IS_APPLE)
#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach_time.h>
// Avoid Mac system header macro leak.
#undef TYPE_BOOL
#endif

#if BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
#include <sys/time.h>
#include <unistd.h>
#endif

namespace tachyon {
namespace base {

class TimeDelta;

namespace time_internal {

template <typename Ratio>
int64_t ToInt64(TimeDelta d, Ratio);
inline int64_t ToInt64(TimeDelta d, std::nano);
inline int64_t ToInt64(TimeDelta d, std::micro);
inline int64_t ToInt64(TimeDelta d, std::milli);
inline int64_t ToInt64(TimeDelta d, std::ratio<1>);
inline int64_t ToInt64(TimeDelta d, std::ratio<60>);
inline int64_t ToInt64(TimeDelta d, std::ratio<3600>);

template <std::intmax_t N>
constexpr TimeDelta FromInt64(int64_t v, std::ratio<1, N>);
constexpr TimeDelta FromInt64(int64_t v, std::ratio<60>);
constexpr TimeDelta FromInt64(int64_t v, std::ratio<3600>);

}  // namespace time_internal

template <typename T>
constexpr TimeDelta Microseconds(T n);

// TimeDelta ------------------------------------------------------------------

class TACHYON_EXPORT TimeDelta {
 public:
  constexpr TimeDelta() = default;

#if BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
  static TimeDelta FromTimeSpec(const timespec& ts);
#endif
#if BUILDFLAG(IS_APPLE)
#if BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)
  static TimeDelta FromMachTime(uint64_t mach_time);
#endif  // BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)
#endif  // BUILDFLAG(IS_APPLE)

  template <typename T>
  static TimeDelta FromChrono(T d);
  static constexpr TimeDelta FromChronoHours(std::chrono::hours d);
  static constexpr TimeDelta FromChronoMinutes(std::chrono::minutes d);
  static constexpr TimeDelta FromChronoSeconds(std::chrono::seconds d);
  static constexpr TimeDelta FromChronoMilliseconds(
      std::chrono::milliseconds d);
  static constexpr TimeDelta FromChronoMicroseconds(
      std::chrono::microseconds d);
  static constexpr TimeDelta FromChronoNanoseconds(std::chrono::nanoseconds d);

  static TimeDelta FromAbsl(absl::Duration d);

  // Converts an integer value representing TimeDelta to a class. This is used
  // when deserializing a |TimeDelta| structure, using a value known to be
  // compatible. It is not provided as a constructor because the integer type
  // may be unclear from the perspective of a caller.
  //
  // DEPRECATED - Do not use in new code. http://crbug.com/634507
  static constexpr TimeDelta FromInternalValue(int64_t delta) {
    return TimeDelta(delta);
  }

  // Returns the maximum time delta, which should be greater than any reasonable
  // time delta we might compare it to. If converted to double with ToDouble()
  // it becomes an IEEE double infinity. Use FiniteMax() if you want a very
  // large number that doesn't do this. TimeDelta math saturates at the end
  // points so adding to TimeDelta::Max() leaves the value unchanged.
  // Subtracting should leave the value unchanged but currently changes it
  // TODO(https://crbug.com/869387).
  static constexpr TimeDelta Max();

  // Returns the minimum time delta, which should be less than than any
  // reasonable time delta we might compare it to. For more details see the
  // comments for Max().
  static constexpr TimeDelta Min();

  // Returns the maximum time delta which is not equivalent to infinity. Only
  // subtracting a finite time delta from this time delta has a defined result.
  static constexpr TimeDelta FiniteMax();

  // Returns the minimum time delta which is not equivalent to -infinity. Only
  // adding a finite time delta to this time delta has a defined result.
  static constexpr TimeDelta FiniteMin();

  // Returns the magnitude (absolute value) of this TimeDelta.
  constexpr TimeDelta magnitude() const { return TimeDelta(delta_.Abs()); }

  // Returns true if the time delta is a zero, positive or negative time delta.
  constexpr bool is_zero() const { return delta_ == 0; }
  constexpr bool is_positive() const { return delta_ > 0; }
  constexpr bool is_negative() const { return delta_ < 0; }

  // Returns true if the time delta is the maximum/minimum time delta.
  constexpr bool is_max() const { return *this == Max(); }
  constexpr bool is_min() const { return *this == Min(); }
  constexpr bool is_inf() const { return is_min() || is_max(); }

#if BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
  struct timespec ToTimeSpec() const;
#endif
  template <typename T>
  T ToChrono() const;
  std::chrono::hours ToChronoHours() const;
  std::chrono::minutes ToChronoMinutes() const;
  std::chrono::seconds ToChronoSeconds() const;
  std::chrono::milliseconds ToChronoMilliseconds() const;
  constexpr std::chrono::microseconds ToChronoMicroseconds() const {
    return std::chrono::microseconds(delta_);
  }
  std::chrono::nanoseconds ToChronoNanoseconds() const;

  constexpr absl::Duration ToAbsl() const {
    return absl::Microseconds(static_cast<int64_t>(delta_));
  }

  // Returns the frequency in Hertz (cycles per second) that has a period of
  // *this.
  constexpr double ToHz() const;

  // Returns the time delta in some unit. Minimum argument values return as
  // -inf for doubles and min type values otherwise. Maximum ones are treated as
  // +inf for doubles and max type values otherwise. Their results will produce
  // an is_min() or is_max() TimeDelta. The InXYZF versions return a floating
  // point value. The InXYZ versions return a truncated value (aka rounded
  // towards zero, std::trunc() behavior). The InXYZFloored() versions round to
  // lesser integers (std::floor() behavior). The XYZRoundedUp() versions round
  // up to greater integers (std::ceil() behavior). WARNING: Floating point
  // arithmetic is such that XXX(t.InXXXF()) may not precisely equal |t|.
  // Hence, floating point values should not be used for storage.
  int InDays() const;
  int InDaysFloored() const;
  constexpr int InHours() const;
  constexpr int InMinutes() const;
  constexpr double InSecondsF() const;
  constexpr int64_t InSeconds() const;
  int64_t InSecondsFloored() const;
  double InMillisecondsF() const;
  int64_t InMilliseconds() const;
  int64_t InMillisecondsRoundedUp() const;
  constexpr int64_t InMicroseconds() const { return delta_; }
  double InMicrosecondsF() const;
  constexpr int64_t InNanoseconds() const;

  // Computations with other deltas.
  constexpr TimeDelta operator+(TimeDelta other) const;
  constexpr TimeDelta operator-(TimeDelta other) const;

  constexpr TimeDelta& operator+=(TimeDelta other) {
    return *this = (*this + other);
  }
  constexpr TimeDelta& operator-=(TimeDelta other) {
    return *this = (*this - other);
  }
  constexpr TimeDelta operator-() const {
    if (!is_inf()) return TimeDelta(-delta_);
    return (delta_ < 0) ? Max() : Min();
  }

  // Computations with numeric types.
  template <typename T>
  constexpr TimeDelta operator*(T a) const {
    return TimeDelta(int64_t{delta_ * a});
  }
  template <typename T>
  constexpr TimeDelta operator/(T a) const {
    return TimeDelta(int64_t{delta_ / a});
  }
  template <typename T>
  constexpr TimeDelta& operator*=(T a) {
    return *this = (*this * a);
  }
  template <typename T>
  constexpr TimeDelta& operator/=(T a) {
    return *this = (*this / a);
  }

  // This does floating-point division. For an integer result, either call
  // IntDiv(), or (possibly clearer) use this operator with
  // tachyon::Clamp{Ceil,Floor,Round}() or tachyon::saturated_cast() (for
  // truncation). Note that converting to double here drops precision to 53
  // bits.
  constexpr double operator/(TimeDelta a) const {
    // 0/0 and inf/inf (any combination of positive and negative) are invalid
    // (they are almost certainly not intentional, and result in NaN, which
    // turns into 0 if clamped to an integer; this makes introducing subtle bugs
    // too easy).
    CHECK(!is_zero() || !a.is_zero());
    CHECK(!is_inf() || !a.is_inf());

    return ToDouble() / a.ToDouble();
  }
  constexpr int64_t IntDiv(TimeDelta a) const {
    if (!is_inf() && !a.is_zero()) return int64_t{delta_ / a.delta_};

    // For consistency, use the same edge case CHECKs and behavior as the code
    // above.
    CHECK(!is_zero() || !a.is_zero());
    CHECK(!is_inf() || !a.is_inf());
    return ((delta_ < 0) == (a.delta_ < 0))
               ? std::numeric_limits<int64_t>::max()
               : std::numeric_limits<int64_t>::min();
  }

  constexpr TimeDelta operator%(TimeDelta a) const {
    return TimeDelta(
        (is_inf() || a.is_zero() || a.is_inf()) ? delta_ : (delta_ % a.delta_));
  }
  constexpr TimeDelta& operator%=(TimeDelta other) {
    return *this = (*this % other);
  }

  // Comparison operators.
  constexpr bool operator==(TimeDelta other) const {
    return delta_ == other.delta_;
  }
  constexpr bool operator!=(TimeDelta other) const {
    return delta_ != other.delta_;
  }
  constexpr bool operator<(TimeDelta other) const {
    return delta_ < other.delta_;
  }
  constexpr bool operator<=(TimeDelta other) const {
    return delta_ <= other.delta_;
  }
  constexpr bool operator>(TimeDelta other) const {
    return delta_ > other.delta_;
  }
  constexpr bool operator>=(TimeDelta other) const {
    return delta_ >= other.delta_;
  }

  // Returns this delta, ceiled/floored/rounded-away-from-zero to the nearest
  // multiple of |interval|.
  TimeDelta CeilToMultiple(TimeDelta interval) const;
  TimeDelta FloorToMultiple(TimeDelta interval) const;
  TimeDelta RoundToMultiple(TimeDelta interval) const;

 private:
  // Constructs a delta given the duration in microseconds. This is private
  // to avoid confusion by callers with an integer constructor. Use
  // tachyon::base::Seconds, tachyon::base::Milliseconds, etc. instead.
  constexpr explicit TimeDelta(int64_t delta_us) : delta_(delta_us) {}
  constexpr explicit TimeDelta(ClampedNumeric<int64_t> delta_us)
      : delta_(delta_us) {}

  // Returns a double representation of this TimeDelta's tick count.  In
  // particular, Max()/Min() are converted to +/-infinity.
  constexpr double ToDouble() const {
    if (!is_inf()) return static_cast<double>(delta_);
    return (delta_ < 0) ? -std::numeric_limits<double>::infinity()
                        : std::numeric_limits<double>::infinity();
  }

  template <typename Ratio>
  friend int64_t time_internal::ToInt64(TimeDelta d, Ratio);
  template <std::intmax_t N>
  friend constexpr TimeDelta time_internal::FromInt64(int64_t v,
                                                      std::ratio<1, N>);
  friend constexpr TimeDelta time_internal::FromInt64(int64_t v,
                                                      std::ratio<60>);
  friend constexpr TimeDelta time_internal::FromInt64(int64_t v,
                                                      std::ratio<3600>);

  // Delta in microseconds.
  ClampedNumeric<int64_t> delta_ = 0;
};

constexpr TimeDelta TimeDelta::operator+(TimeDelta other) const {
  if (!other.is_inf()) return TimeDelta(delta_ + other.delta_);

  // Additions involving two infinities are only valid if signs match.
  CHECK(!is_inf() || (delta_ == other.delta_));
  return other;
}

constexpr TimeDelta TimeDelta::operator-(TimeDelta other) const {
  if (!other.is_inf()) return TimeDelta(delta_ - other.delta_);

  // Subtractions involving two infinities are only valid if signs differ.
  CHECK_NE(int64_t{delta_}, int64_t{other.delta_});
  return (other.delta_ < 0) ? Max() : Min();
}

template <typename T>
constexpr TimeDelta operator*(T a, TimeDelta td) {
  return td * a;
}

// For logging use only.
TACHYON_EXPORT std::ostream& operator<<(std::ostream& os, TimeDelta time_delta);

namespace time_internal {

template <typename Ratio>
int64_t ToInt64(TimeDelta d, Ratio) {
  // Note: This may be used on MSVC, which may have a system_clock period of
  // std::ratio<1, 10 * 1000 * 1000>
  return (d * static_cast<double>(Ratio::den) / Ratio::num).InSeconds();
}
// Fastpath implementations for the 6 common duration units.
int64_t ToInt64(TimeDelta d, std::nano) { return d.InNanoseconds(); }
int64_t ToInt64(TimeDelta d, std::micro) { return d.InMicroseconds(); }
int64_t ToInt64(TimeDelta d, std::milli) { return d.InMilliseconds(); }
int64_t ToInt64(TimeDelta d, std::ratio<1>) { return d.InSeconds(); }
int64_t ToInt64(TimeDelta d, std::ratio<60>) { return d.InMinutes(); }
int64_t ToInt64(TimeDelta d, std::ratio<3600>) { return d.InHours(); }

// TimeBase--------------------------------------------------------------------

// Do not reference the time_internal::TimeBase template class directly.  Please
// use one of the time subclasses instead, and only reference the public
// TimeBase members via those classes.

// Provides value storage and comparison/math operations common to all time
// classes. Each subclass provides for strong type-checking to ensure
// semantically meaningful comparison/math of time values from the same clock
// source or timeline.
template <class TimeClass>
class TimeBase {
 public:
  static constexpr int64_t kHoursPerDay = 24;
  static constexpr int64_t kSecondsPerMinute = 60;
  static constexpr int64_t kMinutesPerHour = 60;
  static constexpr int64_t kSecondsPerHour =
      kSecondsPerMinute * kMinutesPerHour;
  static constexpr int64_t kMillisecondsPerSecond = 1000;
  static constexpr int64_t kMillisecondsPerDay =
      kMillisecondsPerSecond * kSecondsPerHour * kHoursPerDay;
  static constexpr int64_t kMicrosecondsPerMillisecond = 1000;
  static constexpr int64_t kMicrosecondsPerSecond =
      kMicrosecondsPerMillisecond * kMillisecondsPerSecond;
  static constexpr int64_t kMicrosecondsPerMinute =
      kMicrosecondsPerSecond * kSecondsPerMinute;
  static constexpr int64_t kMicrosecondsPerHour =
      kMicrosecondsPerMinute * kMinutesPerHour;
  static constexpr int64_t kMicrosecondsPerDay =
      kMicrosecondsPerHour * kHoursPerDay;
  static constexpr int64_t kMicrosecondsPerWeek = kMicrosecondsPerDay * 7;
  static constexpr int64_t kNanosecondsPerMicrosecond = 1000;
  static constexpr int64_t kNanosecondsPerSecond =
      kNanosecondsPerMicrosecond * kMicrosecondsPerSecond;

  // TODO(https://crbug.com/1392437): Remove concept of "null" from base::Time.
  //
  // Warning: Be careful when writing code that performs math on time values,
  // since it's possible to produce a valid "zero" result that should not be
  // interpreted as a "null" value. If you find yourself using this method or
  // the zero-arg default constructor, please consider using an optional to
  // express the null state.
  //
  // Returns true if this object has not been initialized (probably).
  constexpr bool is_null() const { return us_ == 0; }

  // Returns true if this object represents the maximum/minimum time.
  constexpr bool is_max() const { return *this == Max(); }
  constexpr bool is_min() const { return *this == Min(); }
  constexpr bool is_inf() const { return is_min() || is_max(); }

  // Returns the maximum/minimum times, which should be greater/less than than
  // any reasonable time with which we might compare it.
  static constexpr TimeClass Max() {
    return TimeClass(std::numeric_limits<int64_t>::max());
  }

  static constexpr TimeClass Min() {
    return TimeClass(std::numeric_limits<int64_t>::min());
  }

  // The amount of time since the origin (or "zero") point. This is a syntactic
  // convenience to aid in code readability, mainly for debugging/testing use
  // cases.
  //
  // Warning: While the Time subclass has a fixed origin point, the origin for
  // the other subclasses can vary each time the application is restarted.
  constexpr TimeDelta since_origin() const;

  // Compute the difference between two times.
#if !defined(__aarch64__) && BUILDFLAG(IS_ANDROID)
  NOINLINE  // https://crbug.com/1369775
#endif
      constexpr TimeDelta
      operator-(const TimeBase<TimeClass>& other) const;

  // Return a new time modified by some delta.
  constexpr TimeClass operator+(TimeDelta delta) const;
  constexpr TimeClass operator-(TimeDelta delta) const;

  // Modify by some time delta.
  constexpr TimeClass& operator+=(TimeDelta delta) {
    return static_cast<TimeClass&>(*this = (*this + delta));
  }
  constexpr TimeClass& operator-=(TimeDelta delta) {
    return static_cast<TimeClass&>(*this = (*this - delta));
  }

  // Comparison operators
  constexpr bool operator==(const TimeBase<TimeClass>& other) const {
    return us_ == other.us_;
  }
  constexpr bool operator!=(const TimeBase<TimeClass>& other) const {
    return us_ != other.us_;
  }
  constexpr bool operator<(const TimeBase<TimeClass>& other) const {
    return us_ < other.us_;
  }
  constexpr bool operator<=(const TimeBase<TimeClass>& other) const {
    return us_ <= other.us_;
  }
  constexpr bool operator>(const TimeBase<TimeClass>& other) const {
    return us_ > other.us_;
  }
  constexpr bool operator>=(const TimeBase<TimeClass>& other) const {
    return us_ >= other.us_;
  }

 protected:
  constexpr explicit TimeBase(int64_t us) : us_(us) {}

  // Time value in a microsecond timebase.
  ClampedNumeric<int64_t> us_;
};

}  // namespace time_internal

template <class TimeClass>
inline constexpr TimeClass operator+(TimeDelta delta, TimeClass t) {
  return t + delta;
}

// Time -----------------------------------------------------------------------

// Represents a wall clock time in UTC. Values are not guaranteed to be
// monotonically non-decreasing and are subject to large amounts of skew.

// If you measure time with <chrono> then it looks like below.
//
// #include <chrono>
// #include <ratio>
//
// std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
// std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
// std::chrono::duration<double, std::micro> delta = t2 - t1;
// double us = delta.count();
//
// Using Time looks like below which is more simple.
//
// #include "tachyon/base/time/time.h"
//
// tachyon::base::Time t1 = tachyon::base::Time::Now();
// tachyon::base::Time t2 = tachyon::base::Time::Now();
// tachyon::base::TimeDelta delta = t2 -t1;
// double us = delta.InMicrosecondsF();
class TACHYON_EXPORT Time : public time_internal::TimeBase<Time> {
 public:
  constexpr Time() : TimeBase(0) {}

  // Returns the current time. Watch out, the system might adjust its clock
  // in which case time will actually go backwards. We don't guarantee that
  // times are increasing, or that two calls to Now() won't be the same.
  // You can think of it as std::chrono::system_clock::now(). It delegates
  // to call absl::Now().
  static Time Now();

  // Converts to/from TimeDeltas relative to the Unix epoch (1970-01-01
  // 00:00:00 UTC).
  //
  //   // Serialization:
  //   tachyon::base::Time last_updated = ...;
  //   SaveToDatabase(last_updated.ToDeltaSinceUnixEpoch().InMicroseconds());
  //
  //   // Deserialization:
  //   tachyon::base::Time last_updated =
  //   tachyon::base::Time::FromDeltaSinceUnixEpoch(
  //       tachyon::base::Microseconds(LoadFromDatabase()));
  static constexpr Time FromDeltaSinceUnixEpoch(TimeDelta delta) {
    return Time(delta.InMicroseconds());
  }
  constexpr TimeDelta ToDeltaSinceUnixEpoch() const {
    return Microseconds(us_);
  }

  // Converts to/from time_t in UTC and a Time class.
  static constexpr Time FromTimeT(time_t tt);
  time_t ToTimeT() const;

  // Converts time to/from a double which is the number of seconds since epoch
  // (Jan 1, 1970).  Webkit uses this format to represent time.
  // Because WebKit initializes double time value to 0 to indicate "not
  // initialized", we map it to empty Time object that also means "not
  // initialized".
  static Time FromDoubleT(double dt);
  double ToDoubleT() const;

#if BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
  // Converts the timespec structure to time. MacOS X 10.8.3 (and tentatively,
  // earlier versions) will have the |ts|'s tv_nsec component zeroed out,
  // having a 1 second resolution, which agrees with
  // https://developer.apple.com/legacy/library/#technotes/tn/tn1150.html#HFSPlusDates.
  static Time FromTimeSpec(const timespec& ts);

  static Time FromTimeVal(struct timeval t);
  struct timeval ToTimeVal() const;
#endif

  static Time FromAbsl(absl::Time time);
  absl::Time ToAbsl() const;

  static Time FromChrono(std::chrono::system_clock::time_point tp);
  std::chrono::system_clock::time_point ToChrono() const;

 private:
  friend class time_internal::TimeBase<Time>;

  // NOTE: chromium's time is calculated from windows FILETIME epoch
  // (1601-01-01 00:00:00 UTC). But tachyon currently runs on POSIX,
  // so we calculate from UNIX epoch (1970-01-01 00:00:00 UTC).
  constexpr explicit Time(int64_t microseconds_since_unix_epoch)
      : TimeBase(microseconds_since_unix_epoch) {}
};

// Factory methods that return a TimeDelta of the given unit.
// WARNING: Floating point arithmetic is such that XXX(t.InXXXF()) may not
// precisely equal |t|. Hence, floating point values should not be used for
// storage.

template <typename T>
constexpr TimeDelta Days(T n) {
  return TimeDelta::FromInternalValue(MakeClampedNum(n) *
                                      Time::kMicrosecondsPerDay);
}
template <typename T>
constexpr TimeDelta Hours(T n) {
  return TimeDelta::FromInternalValue(MakeClampedNum(n) *
                                      Time::kMicrosecondsPerHour);
}
template <typename T>
constexpr TimeDelta Minutes(T n) {
  return TimeDelta::FromInternalValue(MakeClampedNum(n) *
                                      Time::kMicrosecondsPerMinute);
}
template <typename T>
constexpr TimeDelta Seconds(T n) {
  return TimeDelta::FromInternalValue(MakeClampedNum(n) *
                                      Time::kMicrosecondsPerSecond);
}
template <typename T>
constexpr TimeDelta Milliseconds(T n) {
  return TimeDelta::FromInternalValue(MakeClampedNum(n) *
                                      Time::kMicrosecondsPerMillisecond);
}
template <typename T>
constexpr TimeDelta Microseconds(T n) {
  return TimeDelta::FromInternalValue(MakeClampedNum(n));
}
template <typename T>
constexpr TimeDelta Nanoseconds(T n) {
  return TimeDelta::FromInternalValue(MakeClampedNum(n) /
                                      Time::kNanosecondsPerMicrosecond);
}
template <typename T>
constexpr TimeDelta Hertz(T n) {
  return n ? TimeDelta::FromInternalValue(Time::kMicrosecondsPerSecond /
                                          MakeClampedNum(n))
           : TimeDelta::Max();
}

// TimeDelta functions that must appear below the declarations of Time/TimeDelta

constexpr double TimeDelta::ToHz() const { return Seconds(1) / *this; }

namespace time_internal {

template <std::intmax_t N>
constexpr TimeDelta FromInt64(int64_t v, std::ratio<1, N>) {
  static_assert(0 < N && N <= 1000 * 1000 * 1000, "Unsupported ratio");
  return TimeDelta::FromInternalValue(saturated_cast<int64_t>(
      static_cast<double>(v) / N * Time::kMicrosecondsPerSecond));
}

constexpr TimeDelta FromInt64(int64_t v, std::ratio<60>) {
  return TimeDelta::FromInternalValue(
      int64_t{ClampMul(v, Time::kMicrosecondsPerMinute)});
}

constexpr TimeDelta FromInt64(int64_t v, std::ratio<3600>) {
  return TimeDelta::FromInternalValue(
      int64_t{ClampMul(v, Time::kMicrosecondsPerHour)});
}

}  // namespace time_internal

// taken and modified from absl::FromChrono().
// static
template <typename T>
TimeDelta TimeDelta::FromChrono(T d) {
  using Rep = typename T::rep;
  using Period = typename T::period;
  static_assert(absl::time_internal::IsValidRep64<Rep>(0),
                "duration::rep is invalid");
  return time_internal::FromInt64(int64_t{d.count()}, Period{});
}

// static
constexpr TimeDelta TimeDelta::FromChronoHours(std::chrono::hours d) {
  return Hours(d.count());
}

// static
constexpr TimeDelta TimeDelta::FromChronoMinutes(std::chrono::minutes d) {
  return Minutes(d.count());
}

// static
constexpr TimeDelta TimeDelta::FromChronoSeconds(std::chrono::seconds d) {
  return Seconds(d.count());
}

// static
constexpr TimeDelta TimeDelta::FromChronoMilliseconds(
    std::chrono::milliseconds d) {
  return Milliseconds(d.count());
}

// static
constexpr TimeDelta TimeDelta::FromChronoMicroseconds(
    std::chrono::microseconds d) {
  return Microseconds(d.count());
}

// static
constexpr TimeDelta TimeDelta::FromChronoNanoseconds(
    std::chrono::nanoseconds d) {
  return Nanoseconds(d.count());
}

// taken and modified from absl::time_internal::ToChronoDuration().
template <typename T>
T TimeDelta::ToChrono() const {
  using Rep = typename T::rep;
  using Period = typename T::period;
  static_assert(absl::time_internal::IsValidRep64<Rep>(0),
                "duration::rep is invalid");
  const auto v = time_internal::ToInt64(*this, Period{});
  if (v > (std::numeric_limits<Rep>::max)()) return (T::max)();
  if (v < (std::numeric_limits<Rep>::min)()) return (T::min)();
  return T{v};
}

constexpr int TimeDelta::InHours() const {
  // saturated_cast<> is necessary since very large (but still less than
  // min/max) deltas would result in overflow.
  return saturated_cast<int>(delta_ / Time::kMicrosecondsPerHour);
}

constexpr int TimeDelta::InMinutes() const {
  // saturated_cast<> is necessary since very large (but still less than
  // min/max) deltas would result in overflow.
  return saturated_cast<int>(delta_ / Time::kMicrosecondsPerMinute);
}

constexpr double TimeDelta::InSecondsF() const {
  if (!is_inf())
    return static_cast<double>(delta_) / Time::kMicrosecondsPerSecond;
  return (delta_ < 0) ? -std::numeric_limits<double>::infinity()
                      : std::numeric_limits<double>::infinity();
}

constexpr int64_t TimeDelta::InSeconds() const {
  return is_inf() ? delta_ : (delta_ / Time::kMicrosecondsPerSecond);
}

constexpr int64_t TimeDelta::InNanoseconds() const {
  return ClampMul(delta_, Time::kNanosecondsPerMicrosecond);
}

// static
constexpr TimeDelta TimeDelta::Max() {
  return TimeDelta(std::numeric_limits<int64_t>::max());
}

// static
constexpr TimeDelta TimeDelta::Min() {
  return TimeDelta(std::numeric_limits<int64_t>::min());
}

// static
constexpr TimeDelta TimeDelta::FiniteMax() {
  return TimeDelta(std::numeric_limits<int64_t>::max() - 1);
}

// static
constexpr TimeDelta TimeDelta::FiniteMin() {
  return TimeDelta(std::numeric_limits<int64_t>::min() + 1);
}

// TimeBase functions that must appear below the declarations of Time/TimeDelta
namespace time_internal {

template <class TimeClass>
constexpr TimeDelta TimeBase<TimeClass>::since_origin() const {
  return Microseconds(us_);
}

template <class TimeClass>
constexpr TimeDelta TimeBase<TimeClass>::operator-(
    const TimeBase<TimeClass>& other) const {
  return Microseconds(us_ - other.us_);
}

template <class TimeClass>
constexpr TimeClass TimeBase<TimeClass>::operator+(TimeDelta delta) const {
  return TimeClass((Microseconds(us_) + delta).InMicroseconds());
}

template <class TimeClass>
constexpr TimeClass TimeBase<TimeClass>::operator-(TimeDelta delta) const {
  return TimeClass((Microseconds(us_) - delta).InMicroseconds());
}

}  // namespace time_internal

// Time functions that must appear below the declarations of Time/TimeDelta

// static
constexpr Time Time::FromTimeT(time_t tt) {
  if (tt == 0) return Time();  // Preserve 0 so we can tell it doesn't exist.
  return (tt == std::numeric_limits<time_t>::max())
             ? Max()
             : Time::FromDeltaSinceUnixEpoch(Seconds(tt));
}

// For logging use only.
TACHYON_EXPORT std::ostream& operator<<(std::ostream& os, Time time);

// TimeTicks ------------------------------------------------------------------

// Represents monotonically non-decreasing clock time.
//
// If you measure time with <chrono> then it looks like below.
//
// #include <chrono>
// #include <ratio>
//
// std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
// std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
// std::chrono::duration<double, std::micro> delta = t2 - t1;
// double us = delta.count();
//
// Using Time looks like below which is more simple.
//
// #include "tachyon/base/time/time.h"
//
// tachyon::base::TimeTicks t1 = tachyon::base::TimeTicks::Now();
// tachyon::base::TimeTicks t2 = tachyon::base::TimeTicks::Now();
// tachyon::base::TimeDelta delta = t2 -t1;
// double us = delta.InMicrosecondsF();
class TACHYON_EXPORT TimeTicks : public time_internal::TimeBase<TimeTicks> {
 public:
  constexpr TimeTicks() : TimeBase(0) {}

  // Platform-dependent tick count representing "right now." When
  // IsHighResolution() returns false, the resolution of the clock could be
  // as coarse as ~15.6ms. Otherwise, the resolution should be no worse than one
  // microsecond.
  static TimeTicks Now();

  // Returns true if the high resolution clock is working on this system and
  // Now() will return high resolution values. Note that, on systems where the
  // high resolution clock works but is deemed inefficient, the low resolution
  // clock will be used instead.
  [[nodiscard]] static bool IsHighResolution();

  // Returns true if TimeTicks is consistent across processes, meaning that
  // timestamps taken on different processes can be safely compared with one
  // another. (Note that, even on platforms where this returns true, time values
  // from different threads that are within one tick of each other must be
  // considered to have an ambiguous ordering.)
  [[nodiscard]] static bool IsConsistentAcrossProcesses();

#if BUILDFLAG(IS_APPLE)
#if BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)
  static TimeTicks FromMachAbsoluteTime(uint64_t mach_absolute_time);

  // Sets the current Mach timebase to `timebase`. Returns the old timebase.
  static mach_timebase_info_data_t SetMachTimebaseInfoForTesting(
      mach_timebase_info_data_t timebase);

#endif  // BUILDFLAG(ENABLE_MACH_ABSOLUTE_TIME_TICKS)
#endif  // BUILDFLAG(IS_APPLE)

  static TimeTicks FromChrono(std::chrono::steady_clock::time_point tp);
  std::chrono::steady_clock::time_point ToChrono() const;

 private:
  friend class time_internal::TimeBase<TimeTicks>;

  // Please use Now() to create a new object. This is for internal use
  // and testing.
  constexpr explicit TimeTicks(int64_t us) : TimeBase(us) {}
};

// For logging use only.
TACHYON_EXPORT std::ostream& operator<<(std::ostream& os, TimeTicks time_ticks);

// ThreadTicks ----------------------------------------------------------------

// Represents a clock, specific to a particular thread, than runs only while the
// thread is running.
//
// There's no way of using thread ticks by <chrono>.
// Using Time looks like below which is more simple.
//
// #include "tachyon/base/time/time.h"
//
// tachyon::base::ThreadTicks t1 = tachyon::base::ThreadTicks::Now();
// tachyon::base::ThreadTicks t2 = tachyon::base::ThreadTicks::Now();
// tachyon::base::TimeDelta delta = t2 -t1;
// double us = delta.InMicrosecondsF();
class TACHYON_EXPORT ThreadTicks : public time_internal::TimeBase<ThreadTicks> {
 public:
  constexpr ThreadTicks() : TimeBase(0) {}

  // Returns true if ThreadTicks::Now() is supported on this system.
  [[nodiscard]] static bool IsSupported() {
#if (defined(_POSIX_THREAD_CPUTIME) && (_POSIX_THREAD_CPUTIME >= 0)) || \
    BUILDFLAG(IS_APPLE) || BUILDFLAG(IS_ANDROID) || BUILDFLAG(IS_FUCHSIA)
    return true;
#elif BUILDFLAG(IS_WIN)
    return IsSupportedWin();
#else
    return false;
#endif
  }

  // Waits until the initialization is completed. Needs to be guarded with a
  // call to IsSupported().
  static void WaitUntilInitialized() {
#if BUILDFLAG(IS_WIN)
    WaitUntilInitializedWin();
#endif
  }

  // Returns thread-specific CPU-time on systems that support this feature.
  // Needs to be guarded with a call to IsSupported(). Use this timer
  // to (approximately) measure how much time the calling thread spent doing
  // actual work vs. being de-scheduled. May return bogus results if the thread
  // migrates to another CPU between two calls. Returns an empty ThreadTicks
  // object until the initialization is completed. If a clock reading is
  // absolutely needed, call WaitUntilInitialized() before this method.
  static ThreadTicks Now();

 private:
  friend class time_internal::TimeBase<ThreadTicks>;

  // Please use Now() to create a new object. This is for internal use
  // and testing.
  constexpr explicit ThreadTicks(int64_t us) : TimeBase(us) {}
};

// For logging use only.
TACHYON_EXPORT std::ostream& operator<<(std::ostream& os,
                                        ThreadTicks thread_ticks);

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_BASE_TIME_TIME_H_
