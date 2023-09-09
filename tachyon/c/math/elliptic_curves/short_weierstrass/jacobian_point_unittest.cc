#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_prime_field_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::c::math {

namespace {

class JacobianPointTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    tachyon::math::bn254::G1JacobianPoint::Curve::Init();
  }

  void SetUp() override {
    a_ = tachyon::math::bn254::G1JacobianPoint::Random();
    b_ = tachyon::math::bn254::G1JacobianPoint::Random();

    c_a_ = cc::math::ToCJacobianPoint(a_);
    c_b_ = cc::math::ToCJacobianPoint(b_);
  }

 protected:
  tachyon::math::bn254::G1JacobianPoint a_;
  tachyon::math::bn254::G1JacobianPoint b_;
  tachyon_bn254_g1_jacobian c_a_;
  tachyon_bn254_g1_jacobian c_b_;
};

}  // namespace

TEST_F(JacobianPointTest, Zero) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_jacobian_zero();
  EXPECT_TRUE(cc::math::ToJacobianPoint(c_ret).IsZero());
}

TEST_F(JacobianPointTest, Generator) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_jacobian_generator();
  EXPECT_EQ(cc::math::ToJacobianPoint(c_ret),
            tachyon::math::bn254::G1JacobianPoint::Generator());
}

TEST_F(JacobianPointTest, Random) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_jacobian_random();
  EXPECT_NE(cc::math::ToJacobianPoint(c_ret), a_);
}

}  // namespace tachyon::c::math
