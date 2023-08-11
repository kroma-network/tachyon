#include "tachyon/base/time/time_stamp.h"

namespace tachyon::base {

TimeDelta TimeStamp::GetTimeDelta(bool update) {
  TimeTicks now = TimeTicks::Now();
  if (update && base_time_.is_null()) {
    base_time_ = now;
  }

  return now - base_time_;
}

}  // namespace tachyon::base
