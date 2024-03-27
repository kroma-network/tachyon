#include "circomlib/base/g2_affine_point.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"

namespace tachyon::circom {

class G2AffinePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::bn254::G2Curve::Init(); }
};

TEST_F(G2AffinePointTest, Conversions) {
  math::bn254::G2AffinePoint expected = math::bn254::G2AffinePoint::Random();
  {
    G2AffinePoint affine_point = G2AffinePoint::FromNative<true>(expected);
    math::bn254::G2AffinePoint actual =
        affine_point.ToNative<true, math::bn254::G2Curve>();
    EXPECT_EQ(actual, expected);
  }
  {
    G2AffinePoint affine_point = G2AffinePoint::FromNative<false>(expected);
    math::bn254::G2AffinePoint actual =
        affine_point.ToNative<false, math::bn254::G2Curve>();
    EXPECT_EQ(actual, expected);
  }
}

}  // namespace tachyon::circom
