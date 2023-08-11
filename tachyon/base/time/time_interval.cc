#include "tachyon/base/time/time_interval.h"

#include <utility>

namespace tachyon::base {

void TimeInterval::Start() {
#if DCHECK_IS_ON()
  DCHECK(!std::exchange(started_, true));
#endif
  last_time_ = TimeTicks::Now();
}

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
