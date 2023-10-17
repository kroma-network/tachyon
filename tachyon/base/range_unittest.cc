#include "tachyon/base/range.h"

#include "gtest/gtest.h"

namespace tachyon::base {

template <typename RangeType>
class RangeTest : public testing::Test {};

using RangeTypes =
    testing::Types<Range<int, false, false>, Range<int, false, true>,
                   Range<int, true, false>, Range<int, true, true>>;
TYPED_TEST_SUITE(RangeTest, RangeTypes);

TYPED_TEST(RangeTest, IsEmpty) {
  using RangeType = TypeParam;

  RangeType range(3, 3);
  if constexpr (RangeType::kIsStartInclusive && RangeType::kIsEndInclusive) {
    EXPECT_FALSE(range.IsEmpty());
  } else {
    EXPECT_TRUE(range.IsEmpty());
  }

  range = RangeType(3, 4);
  EXPECT_FALSE(range.IsEmpty());
}

TYPED_TEST(RangeTest, Contains) {
  using RangeType = TypeParam;

  RangeType range(3, 4);
  EXPECT_FALSE(range.Contains(2));
  if constexpr (RangeType::kIsStartInclusive) {
    EXPECT_TRUE(range.Contains(3));
  } else {
    EXPECT_FALSE(range.Contains(3));
  }
  if constexpr (RangeType::kIsEndInclusive) {
    EXPECT_TRUE(range.Contains(4));
  } else {
    EXPECT_FALSE(range.Contains(4));
  }
  EXPECT_FALSE(range.Contains(5));
}

}  // namespace tachyon::base
