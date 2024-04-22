#include "tachyon/math/base/egcd.h"

#include "gtest/gtest.h"

#include "tachyon/base/random.h"

namespace tachyon::math {

TEST(EGCDTest, Compute) {
  int x = base::Uniform(base::Range<int>::From(1));
  int y = base::Uniform(base::Range<int>::From(1));
  EGCD<int>::Result result = EGCD<int>::Compute(x, y);
  EXPECT_TRUE(result.IsValid(x, y));
}

}  // namespace tachyon::math
