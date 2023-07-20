#include "tachyon/math/base/sign.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

template <typename IntegerType>
class SignTest : public testing::Test {};

using IntegerTypes = testing::Types<int, unsigned int, double>;
TYPED_TEST_SUITE(SignTest, IntegerTypes);

TYPED_TEST(SignTest, GetSign) {
  EXPECT_EQ(GetSign(TypeParam(0)), Sign::kZero);
  EXPECT_EQ(GetSign(TypeParam(1)), Sign::kPositive);
  if constexpr (std::is_signed_v<TypeParam>) {
    EXPECT_EQ(GetSign(TypeParam(-1)), Sign::kNegative);
  }
  if constexpr (std::is_floating_point_v<TypeParam>) {
    EXPECT_EQ(GetSign(TypeParam(std::numeric_limits<TypeParam>::quiet_NaN())),
              Sign::kNaN);
    EXPECT_EQ(
        GetSign(TypeParam(std::numeric_limits<TypeParam>::signaling_NaN())),
        Sign::kNaN);
    EXPECT_EQ(GetSign(TypeParam(std::numeric_limits<TypeParam>::infinity())),
              Sign::kPositive);
    EXPECT_EQ(GetSign(TypeParam(-std::numeric_limits<TypeParam>::infinity())),
              Sign::kNegative);
  }
}

}  // namespace math
}  // namespace tachyon
