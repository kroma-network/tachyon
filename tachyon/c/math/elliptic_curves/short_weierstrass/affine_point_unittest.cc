#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_test.h"
#include "tachyon/c/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon {

namespace {

class AffinePointTest : public c::math::bn254::G1Test {
 public:
  void SetUp() override {
    a_ = math::bn254::G1AffinePoint::Random();
    b_ = math::bn254::G1AffinePoint::Random();

    c_a_ = c::math::ToCAffinePoint(a_);
    c_b_ = c::math::ToCAffinePoint(b_);
  }

 protected:
  math::bn254::G1AffinePoint a_;
  math::bn254::G1AffinePoint b_;
  tachyon_bn254_g1_affine c_a_;
  tachyon_bn254_g1_affine c_b_;
};

}  // namespace

TEST_F(AffinePointTest, Zero) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_zero();
  EXPECT_TRUE(c::math::ToAffinePoint(c_ret).IsZero());
}

TEST_F(AffinePointTest, Generator) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_generator();
  EXPECT_EQ(c::math::ToAffinePoint(c_ret),
            math::bn254::G1AffinePoint::Generator());
}

TEST_F(AffinePointTest, Random) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_random();
  EXPECT_NE(c::math::ToAffinePoint(c_ret), a_);
}

TEST_F(AffinePointTest, Eq) {
  EXPECT_EQ(tachyon_bn254_g1_affine_eq(&c_a_, &c_b_), a_ == b_);
}

TEST_F(AffinePointTest, Ne) {
  EXPECT_EQ(tachyon_bn254_g1_affine_ne(&c_a_, &c_b_), a_ != b_);
}

TEST_F(AffinePointTest, Add) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_affine_add(&c_a_, &c_b_);
  EXPECT_EQ(c::math::ToJacobianPoint(c_ret), a_ + b_);
}

TEST_F(AffinePointTest, Sub) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_affine_sub(&c_a_, &c_b_);
  EXPECT_EQ(c::math::ToJacobianPoint(c_ret), a_ - b_);
}

TEST_F(AffinePointTest, Neg) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_neg(&c_a_);
  EXPECT_EQ(c::math::ToAffinePoint(c_ret), -a_);
}

TEST_F(AffinePointTest, Dbl) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_affine_dbl(&c_a_);
  EXPECT_EQ(c::math::ToJacobianPoint(c_ret), a_.Double());
}

}  // namespace tachyon
