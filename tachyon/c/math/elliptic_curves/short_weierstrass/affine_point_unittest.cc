#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_prime_field_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/point_test.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"

namespace tachyon {

namespace {

class AffinePointTest : public c::math::bn254::PointTest {
 public:
  void SetUp() override {
    a1_ = math::bn254::G1AffinePoint::Random();
    b1_ = math::bn254::G1AffinePoint::Random();
    a2_ = math::bn254::G1AffinePoint::Random();
    b2_ = math::bn254::G1AffinePoint::Random();

    c_a_ = cc::math::ToCAffinePoint(a_);
    c_b_ = cc::math::ToCAffinePoint(b_);
  }

 protected:
  math::bn254::G1AffinePoint a1_;
  math::bn254::G1AffinePoint b1_;
  math::bn254::G2AffinePoint a2_;
  math::bn254::G2AffinePoint b2_;
  tachyon_bn254_g1_affine c_a1_;
  tachyon_bn254_g1_affine c_b1_;
  tachyon_bn254_g2_affine c_a2_;
  tachyon_bn254_g2_affine c_b2_;
};

}  // namespace

TEST_F(AffinePointTest, Zero) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_zero();
  EXPECT_TRUE(cc::math::ToAffinePoint(c_ret).IsZero());
}

TEST_F(AffinePointTest, Generator) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_generator();
  EXPECT_EQ(cc::math::ToAffinePoint(c_ret),
            math::bn254::G1AffinePoint::Generator());
}

TEST_F(AffinePointTest, Random) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_random();
  EXPECT_NE(cc::math::ToAffinePoint(c_ret), a_);
}

TEST_F(AffinePointTest, Eq) {
  EXPECT_EQ(tachyon_bn254_g1_affine_eq(&c_a_, &c_b_), a_ == b_);
}

TEST_F(AffinePointTest, Ne) {
  EXPECT_EQ(tachyon_bn254_g1_affine_ne(&c_a_, &c_b_), a_ != b_);
}

TEST_F(AffinePointTest, Add) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_affine_add(&c_a_, &c_b_);
  EXPECT_EQ(cc::math::ToJacobianPoint(c_ret), a_ + b_);
}

TEST_F(AffinePointTest, Sub) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_affine_sub(&c_a_, &c_b_);
  EXPECT_EQ(cc::math::ToJacobianPoint(c_ret), a_ - b_);
}

TEST_F(AffinePointTest, Neg) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_neg(&c_a_);
  EXPECT_EQ(cc::math::ToAffinePoint(c_ret), -a_);
}

TEST_F(AffinePointTest, Dbl) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_affine_dbl(&c_a_);
  EXPECT_EQ(cc::math::ToJacobianPoint(c_ret), a_.Double());
}

}  // namespace tachyon
