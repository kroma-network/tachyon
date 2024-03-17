#include "circomlib/base/g1_affine_point.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::circom {

class G1AffinePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

TEST_F(G1AffinePointTest, Conversions) {
  math::bn254::G1AffinePoint expected = math::bn254::G1AffinePoint::Random();
  {
    G1AffinePoint affine_point = G1AffinePoint::FromNative<true>(expected);
    math::bn254::G1AffinePoint actual =
        affine_point.ToNative<true, math::bn254::G1Curve>();
    EXPECT_EQ(actual, expected);
  }
  {
    G1AffinePoint affine_point = G1AffinePoint::FromNative<false>(expected);
    math::bn254::G1AffinePoint actual =
        affine_point.ToNative<false, math::bn254::G1Curve>();
    EXPECT_EQ(actual, expected);
  }
}

}  // namespace tachyon::circom
