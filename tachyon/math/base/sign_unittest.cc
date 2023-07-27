#include "tachyon/math/base/sign.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

template <typename NumType>
class SignTest : public testing::Test {};

using NumTypes = testing::Types<int, unsigned int, double>;
TYPED_TEST_SUITE(SignTest, NumTypes);

TYPED_TEST(SignTest, GetSign) {
  using NumType = TypeParam;

  EXPECT_EQ(GetSign(NumType(0)), Sign::kZero);
  EXPECT_EQ(GetSign(NumType(1)), Sign::kPositive);
  if constexpr (std::is_signed_v<NumType>) {
    EXPECT_EQ(GetSign(NumType(-1)), Sign::kNegative);
  }
  if constexpr (std::is_floating_point_v<NumType>) {
    EXPECT_EQ(GetSign(NumType(std::numeric_limits<NumType>::quiet_NaN())),
              Sign::kNaN);
    EXPECT_EQ(GetSign(NumType(std::numeric_limits<NumType>::signaling_NaN())),
              Sign::kNaN);
    EXPECT_EQ(GetSign(NumType(std::numeric_limits<NumType>::infinity())),
              Sign::kPositive);
    EXPECT_EQ(GetSign(NumType(-std::numeric_limits<NumType>::infinity())),
              Sign::kNegative);
  }
}

}  // namespace math
}  // namespace tachyon
