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
  static void SetUpTestSuite() {
    tachyon::math::bn254::G1ProjectivePoint::Curve::Init();
  }

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

}  // namespace tachyon::c::math
