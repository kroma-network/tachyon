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
  G2AffinePoint affine_point = G2AffinePoint::FromNative(expected);
  EXPECT_EQ(affine_point.ToNative<math::bn254::G2Curve>(), expected);
}

}  // namespace tachyon::circom
