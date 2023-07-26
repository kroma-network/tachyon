#include "tachyon/base/time/time_interval.h"

#include "gtest/gtest.h"

#include "tachyon/base/threading/platform_thread.h"

namespace tachyon {
namespace base {

TEST(TimeIntervalTest, GetTimeDelta) {
  TimeInterval ti;
  EXPECT_EQ(ti.GetTimeDelta(), TimeDelta());
  TimeTicks last_time = ti.last_time_;
  PlatformThread::Sleep(Milliseconds(10));
  EXPECT_GT(ti.GetTimeDelta(), TimeDelta());
  EXPECT_GT(ti.last_time_, last_time);
  last_time = ti.last_time_;
  TimeDelta dt;
  for (int i = 0; i < 3; ++i) {
    PlatformThread::Sleep(Milliseconds(10));
    TimeDelta dt_temp = ti.GetTimeDelta(false);
    EXPECT_GT(dt_temp, dt);
    EXPECT_EQ(ti.last_time_, last_time);
    dt = dt_temp;
  }

  TimeInterval ti2(TimeTicks::Now());
  last_time = ti2.last_time_;
  PlatformThread::Sleep(Milliseconds(10));
  EXPECT_GT(ti2.GetTimeDelta(), TimeDelta());
  EXPECT_GT(ti2.last_time_, last_time);
}

}  // namespace base
}  // namespace tachyon
