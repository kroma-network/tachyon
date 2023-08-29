#include "tachyon/base/time/time_interval.h"

namespace tachyon::base {

void TimeInterval::Reset() { last_time_ = TimeTicks::Now(); }

TimeDelta TimeInterval::GetTimeDelta(bool update) {
  TimeTicks now = TimeTicks::Now();
  TimeDelta dt;
  if (!last_time_.is_null()) {
    dt = now - last_time_;
  }
  if (update) {
    last_time_ = now;
  }
  return dt;
}

}  // namespace tachyon::base
