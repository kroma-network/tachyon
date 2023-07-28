#include "tachyon/base/time/time_stamp.h"

#include "gtest/gtest.h"

#include "tachyon/base/threading/platform_thread.h"

namespace tachyon::base {

TEST(TimeStampTest, GetTimeDelta) {
  TimeStamp ts;
  TimeDelta dt = ts.GetTimeDelta();
  EXPECT_EQ(dt, TimeDelta());
  TimeTicks base_time = ts.base_time_;
  PlatformThread::Sleep(Milliseconds(10));
  EXPECT_GT(ts.GetTimeDelta(), dt);
  EXPECT_EQ(ts.base_time_, base_time);

  TimeStamp ts2;
  dt = ts2.GetTimeDelta(false);
  EXPECT_GT(dt, TimeDelta());
  PlatformThread::Sleep(Milliseconds(10));
  EXPECT_GT(ts2.GetTimeDelta(false), dt);
}

}  // namespace tachyon::base
