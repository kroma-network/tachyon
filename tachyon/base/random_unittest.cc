
#include "tachyon/base/random.h"

#include "gtest/gtest.h"

namespace tachyon::base {

constexpr int kMin = 0;
constexpr int kMax = 1000;
constexpr size_t kCount = 1000;

template <typename RangeType>
class RandomWithRangeTest : public testing::Test {};

using RangeTypes =
    testing::Types<Range<int, false, false>, Range<int, false, true>,
                   Range<int, true, false>, Range<int, true, true>>;
TYPED_TEST_SUITE(RandomWithRangeTest, RangeTypes);

TYPED_TEST(RandomWithRangeTest, SampleDifferentValue) {
  using RangeType = TypeParam;

  int r = Uniform(RangeType(kMin, kMax));
  for (size_t i = 0; i < kCount; ++i) {
    if (r != Uniform(RangeType(kMin, kMax))) {
      SUCCEED();
      return;
    }
  }
  FAIL() << "random seems not working";
}

TYPED_TEST(RandomWithRangeTest, SampleWithinRange) {
  using RangeType = TypeParam;

  RangeType range(kMin, kMax);
  int value = Uniform(range);
  EXPECT_TRUE(range.Contains(value));
}

TEST(RandomTest, UniformElementWithArray) {
  int arr[] = {1, 2, 3};
  int r = UniformElement(arr);
  EXPECT_GE(r, 1);
  EXPECT_LE(r, 3);
}

TEST(RandomTest, UniformElementWithVector) {
  std::vector<int> vec = {1, 2, 3};
  int& r = UniformElement(vec);
  EXPECT_GE(r, 1);
  EXPECT_LE(r, 3);
  r = 2;
}

TEST(RandomTest, UniformElementWithConstVector) {
  const std::vector<int> vec = {1, 2, 3};
  const int& r = UniformElement(vec);
  EXPECT_GE(r, 1);
  EXPECT_LE(r, 3);
}

}  // namespace tachyon::base
