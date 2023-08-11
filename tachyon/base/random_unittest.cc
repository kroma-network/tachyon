
#include "tachyon/base/random.h"

#include "gtest/gtest.h"

namespace tachyon::base {

constexpr const int kMin = 0;
constexpr const int kMax = 1000;
constexpr const size_t kCount = 1000;

TEST(Random, BasicTest) {
  int r = Uniform(kMin, kMax);
  for (size_t i = 0; i < kCount; ++i) {
    if (r != Uniform(kMin, kMax)) {
      SUCCEED();
      return;
    }
  }
  FAIL() << "random seems not working";
}

}  // namespace tachyon::base
