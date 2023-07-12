#include "tachyon/math/elliptic_curves/msm/glv.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls/bls12_381/curve_config.h"

namespace tachyon {
namespace math {

namespace {

class GLVTest : public ::testing::Test {
 public:
  GLVTest() { bls12_381::CurveConfig::Init(); }
  GLVTest(const GLVTest&) = delete;
  GLVTest& operator=(const GLVTest&) = delete;
  ~GLVTest() override = default;
};

}  // namespace

TEST_F(GLVTest, Mul) {
  bls12_381::G1JacobianPoint base = bls12_381::G1JacobianPoint::Random();
  bls12_381::Fr scalar = bls12_381::Fr::Random();

  EXPECT_EQ(GLV<bls12_381::CurveConfig>::Mul(base, scalar),
            base.ScalarMul(scalar.ToMpzClass()));

  EXPECT_EQ(GLV<bls12_381::CurveConfig>::Mul(base.ToAffine(), scalar),
            base.ScalarMul(scalar.ToMpzClass()));
}

}  // namespace math
}  // namespace tachyon
