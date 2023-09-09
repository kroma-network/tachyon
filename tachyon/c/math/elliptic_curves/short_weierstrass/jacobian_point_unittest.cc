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

TEST_F(JacobianPointTest, Eq) {
  EXPECT_EQ(tachyon_bn254_g1_jacobian_eq(&c_a_, &c_b_), a_ == b_);
}

TEST_F(JacobianPointTest, Ne) {
  EXPECT_EQ(tachyon_bn254_g1_jacobian_ne(&c_a_, &c_b_), a_ != b_);
}

TEST_F(JacobianPointTest, Add) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_jacobian_add(&c_a_, &c_b_);
  EXPECT_EQ(cc::math::ToJacobianPoint(c_ret), a_ + b_);

  tachyon::math::bn254::G1AffinePoint d =
      tachyon::math::bn254::G1AffinePoint::Random();
  tachyon_bn254_g1_affine c_d = cc::math::ToCAffinePoint(d);
  c_ret = tachyon_bn254_g1_jacobian_add_mixed(&c_a_, &c_d);
  EXPECT_EQ(cc::math::ToJacobianPoint(c_ret), a_ + d);
}

TEST_F(JacobianPointTest, Sub) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_jacobian_sub(&c_a_, &c_b_);
  EXPECT_EQ(cc::math::ToJacobianPoint(c_ret), a_ - b_);

  tachyon::math::bn254::G1AffinePoint d =
      tachyon::math::bn254::G1AffinePoint::Random();
  tachyon_bn254_g1_affine c_d = cc::math::ToCAffinePoint(d);
  c_ret = tachyon_bn254_g1_jacobian_sub_mixed(&c_a_, &c_d);
  EXPECT_EQ(cc::math::ToJacobianPoint(c_ret), a_ - d);
}

}  // namespace tachyon::c::math
