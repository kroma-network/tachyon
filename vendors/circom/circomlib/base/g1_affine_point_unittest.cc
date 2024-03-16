#include "circomlib/base/g1_affine_point.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::circom {

class G1AffinePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

TEST_F(G1AffinePointTest, Conversions) {
  math::bn254::G1AffinePoint expected = math::bn254::G1AffinePoint::Random();
  G1AffinePoint affine_point = G1AffinePoint::FromNative(expected);
  EXPECT_EQ(affine_point.ToNative<math::bn254::G1Curve>(), expected);
}

}  // namespace tachyon::circom
