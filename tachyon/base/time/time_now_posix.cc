// Copyright (c) 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <time.h>

#include "tachyon/base/numerics/checked_math.h"
#include "tachyon/base/time/time.h"
#include "tachyon/build/build_config.h"

namespace tachyon::base {

namespace {

int64_t ConvertTimespecToMicros(const struct timespec& ts) {
  // On 32-bit systems, the calculation cannot overflow int64_t.
  // 2**32 * 1000000 + 2**64 / 1000 < 2**63
  if (sizeof(ts.tv_sec) <= 4 && sizeof(ts.tv_nsec) <= 8) {
    int64_t result = ts.tv_sec;
    result *= Time::kMicrosecondsPerSecond;
    result += (ts.tv_nsec / Time::kNanosecondsPerMicrosecond);
    return result;
  }
  CheckedNumeric<int64_t> result(ts.tv_sec);
  result *= Time::kMicrosecondsPerSecond;
  result += (ts.tv_nsec / Time::kNanosecondsPerMicrosecond);
  return result.ValueOrDie();
}

// Helper function to get results from clock_gettime() and convert to a
// microsecond timebase. Minimum requirement is MONOTONIC_CLOCK to be supported
// on the system. FreeBSD 6 has CLOCK_MONOTONIC but defines
// _POSIX_MONOTONIC_CLOCK to -1.
#if (BUILDFLAG(IS_POSIX) && defined(_POSIX_MONOTONIC_CLOCK) && \
     _POSIX_MONOTONIC_CLOCK >= 0) ||                           \
    BUILDFLAG(IS_BSD) || BUILDFLAG(IS_ANDROID)
int64_t ClockNow(clockid_t clk_id) {
  struct timespec ts;
  CHECK(clock_gettime(clk_id, &ts) == 0);
  return ConvertTimespecToMicros(ts);
}
#else  // _POSIX_MONOTONIC_CLOCK
#error No usable tick clock function on this platform.
#endif  // _POSIX_MONOTONIC_CLOCK

}  // namespace

// static
TimeTicks TimeTicks::Now() { return TimeTicks(ClockNow(CLOCK_MONOTONIC)); }

// static
ThreadTicks ThreadTicks::Now() {
  return ThreadTicks(ClockNow(CLOCK_THREAD_CPUTIME_ID));
}

}  // namespace tachyon::base
