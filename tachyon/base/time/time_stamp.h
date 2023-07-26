#ifndef TACHYON_BASE_TIME_TIME_STAMP_H_
#define TACHYON_BASE_TIME_TIME_STAMP_H_

#include "gtest/gtest_prod.h"

#include "tachyon/base/time/time.h"
#include "tachyon/export.h"

namespace tachyon {
namespace base {

// This is a convenience class for getting tachyon::base::TimeDelta from a base
// event. A typical usecase is getting timestamp for video frame. In this case,
// Timestamp of Initial frame will be 0 and get increased as video get streamed.
// Another typical usecase is getting timestamp to measure message delivery
// latency between sender and receiver. In this case, you should keep in mind
// that you need to call `GetTimeDelta(/*update=*/false)` so that it doesn't
// update its internal state. NOTE: This class doesn't guarantee thread safety.
//
// Example:
//
//   tachyon::base::TimeStamp ts;
//   while (true) {
//     // heavy calculation
//     tachyon::base::TimeDelta dt = ts.GetTimeDelta();
//   }
//
//   // This is same with below.
//   tachyon::base::TimeTicks base_time;
//   while (true) {
//     // heavy calculation
//     tachyon::base::TimeDelta dt;
//     if (base_time.is_null()) {
//       // First dt is evaluated to zero on purpose.
//       base_time = tachyon::base::TimeTicks::Now();
//     } else {
//       dt = tachyon::base::TimeTicks::Now() - base_time;
//     }
//   }
class TACHYON_EXPORT TimeStamp {
 public:
  constexpr TimeStamp() {}

  // Returns TimeDelta from |base_time_|. For the first call, it returns 0 if
  // you set |update| is true. Otherwise it returns TimeDelta from
  // TimeTicks::Now() regardless of |base_time_|.
  TimeDelta GetTimeDelta(bool update = true);

 private:
  FRIEND_TEST(TimeStampTest, GetTimeDelta);

  TimeTicks base_time_;
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_BASE_TIME_TIME_STAMP_H_
