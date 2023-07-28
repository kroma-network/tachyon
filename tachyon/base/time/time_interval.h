#ifndef TACHYON_BASE_TIME_TIME_INTERVAL_H_
#define TACHYON_BASE_TIME_TIME_INTERVAL_H_

#include "gtest/gtest_prod.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/time/time.h"
#include "tachyon/export.h"

namespace tachyon::base {

// This is a convenience class for getting tachyon::base::TimeDelta between
// events. A typical usecase is getting time delta between robot movements.
// NOTE: This class doesn't guarantee thread safety.
//
// Example:
//
//   tachyon::base::TimeInterval ti(TimeTicks::Now());
//   // or you can do like below.
//   // tachyon::base::TimeInterval ti;
//   // ti.Start();
//   while (true) {
//     // heavy calculation
//     tachyon::base::TimeDelta dt = ti.GetTimeDelta();
//   }
//
//   // This is same with below.
//   tachyon::base::TimeTicks last_time = TimeTicks::Now();
//   while (true) {
//     // heavy calculation
//     tachyon::base::TimeTicks now = tachyon::base::TimeTicks::Now();
//     tachyon::base::TimeDelta dt = now - last_time;
//     last_time = now;
//   }
class TACHYON_EXPORT TimeInterval {
 public:
  constexpr TimeInterval() {}
  explicit constexpr TimeInterval(TimeTicks last_time)
      : last_time_(last_time)
#if DCHECK_IS_ON()
        ,
        started_(true)
#endif
  {
  }

  void Start();

  // Returns TimeDelta from |last_time_|. For the first call, it might return a
  // bogus value if |last_time_| is not given to this class. If |update| is not
  // set, |last_time_| will not be updated.
  TimeDelta GetTimeDelta(bool update = true);

 private:
  FRIEND_TEST(TimeIntervalTest, GetTimeDelta);

  TimeTicks last_time_;
#if DCHECK_IS_ON()
  bool started_ = false;
#endif
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_TIME_TIME_INTERVAL_H_
