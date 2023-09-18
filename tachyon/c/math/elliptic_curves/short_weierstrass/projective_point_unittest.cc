#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_prime_field_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::c::math {

namespace {

class ProjectivePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { tachyon_bn254_g1_init(); }

  void SetUp() override {
    a_ = tachyon::math::bn254::G1ProjectivePoint::Random();
    b_ = tachyon::math::bn254::G1ProjectivePoint::Random();

    c_a_ = cc::math::ToCProjectivePoint(a_);
    c_b_ = cc::math::ToCProjectivePoint(b_);
  }

 protected:
  tachyon::math::bn254::G1ProjectivePoint a_;
  tachyon::math::bn254::G1ProjectivePoint b_;
  tachyon_bn254_g1_projective c_a_;
  tachyon_bn254_g1_projective c_b_;
};

}  // namespace

TEST_F(ProjectivePointTest, Zero) {
  tachyon_bn254_g1_projective c_ret = tachyon_bn254_g1_projective_zero();
  EXPECT_TRUE(cc::math::ToProjectivePoint(c_ret).IsZero());
}

TEST_F(ProjectivePointTest, Generator) {
  tachyon_bn254_g1_projective c_ret = tachyon_bn254_g1_projective_generator();
  EXPECT_EQ(cc::math::ToProjectivePoint(c_ret),
            tachyon::math::bn254::G1ProjectivePoint::Generator());
}

TEST_F(ProjectivePointTest, Random) {
  tachyon_bn254_g1_projective c_ret = tachyon_bn254_g1_projective_random();
  EXPECT_NE(cc::math::ToProjectivePoint(c_ret), a_);
}

TEST_F(ProjectivePointTest, Eq) {
  EXPECT_EQ(tachyon_bn254_g1_projective_eq(&c_a_, &c_b_), a_ == b_);
}

TEST_F(ProjectivePointTest, Ne) {
  EXPECT_EQ(tachyon_bn254_g1_projective_ne(&c_a_, &c_b_), a_ != b_);
}

TEST_F(ProjectivePointTest, Add) {
  tachyon_bn254_g1_projective c_ret =
      tachyon_bn254_g1_projective_add(&c_a_, &c_b_);
  EXPECT_EQ(cc::math::ToProjectivePoint(c_ret), a_ + b_);

  tachyon::math::bn254::G1AffinePoint d =
      tachyon::math::bn254::G1AffinePoint::Random();
  tachyon_bn254_g1_affine c_d = cc::math::ToCAffinePoint(d);
  c_ret = tachyon_bn254_g1_projective_add_mixed(&c_a_, &c_d);
  EXPECT_EQ(cc::math::ToProjectivePoint(c_ret), a_ + d);
}

TEST_F(ProjectivePointTest, Sub) {
  tachyon_bn254_g1_projective c_ret =
      tachyon_bn254_g1_projective_sub(&c_a_, &c_b_);
  EXPECT_EQ(cc::math::ToProjectivePoint(c_ret), a_ - b_);

  tachyon::math::bn254::G1AffinePoint d =
      tachyon::math::bn254::G1AffinePoint::Random();
  tachyon_bn254_g1_affine c_d = cc::math::ToCAffinePoint(d);
  c_ret = tachyon_bn254_g1_projective_sub_mixed(&c_a_, &c_d);
  EXPECT_EQ(cc::math::ToProjectivePoint(c_ret), a_ - d);
}

TEST_F(ProjectivePointTest, Neg) {
  tachyon_bn254_g1_projective c_ret = tachyon_bn254_g1_projective_neg(&c_a_);
  EXPECT_EQ(cc::math::ToProjectivePoint(c_ret), -a_);
}

TEST_F(ProjectivePointTest, Dbl) {
  tachyon_bn254_g1_projective c_ret = tachyon_bn254_g1_projective_dbl(&c_a_);
  EXPECT_EQ(cc::math::ToProjectivePoint(c_ret), a_.Double());
}

}  // namespace tachyon::c::math
