#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2_point_type_traits.h"

namespace tachyon {

namespace {

class AffinePointTest : public testing::Test {
 public:
  void SetUp() override {
    tachyon_bn254_g1_init();
    tachyon_bn254_g2_init();

    g1_a_ = math::bn254::G1AffinePoint::Random();
    g1_b_ = math::bn254::G1AffinePoint::Random();
    g1_c_a_ = c::base::c_cast(g1_a_);
    g1_c_b_ = c::base::c_cast(g1_b_);

    g2_a_ = math::bn254::G2AffinePoint::Random();
    g2_b_ = math::bn254::G2AffinePoint::Random();
    g2_c_a_ = c::base::c_cast(g2_a_);
    g2_c_b_ = c::base::c_cast(g2_b_);
  }

 protected:
  math::bn254::G1AffinePoint g1_a_;
  math::bn254::G1AffinePoint g1_b_;
  tachyon_bn254_g1_affine g1_c_a_;
  tachyon_bn254_g1_affine g1_c_b_;

  math::bn254::G2AffinePoint g2_a_;
  math::bn254::G2AffinePoint g2_b_;
  tachyon_bn254_g2_affine g2_c_a_;
  tachyon_bn254_g2_affine g2_c_b_;
};

}  // namespace

TEST_F(AffinePointTest, Zero) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_zero();
  EXPECT_TRUE(c::base::native_cast(c_ret).IsZero());

  tachyon_bn254_g2_affine c_ret2 = tachyon_bn254_g2_affine_zero();
  EXPECT_TRUE(c::base::native_cast(c_ret2).IsZero());
}

TEST_F(AffinePointTest, Generator) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_generator();
  EXPECT_EQ(c::base::native_cast(c_ret),
            math::bn254::G1AffinePoint::Generator());

  tachyon_bn254_g2_affine c_ret2 = tachyon_bn254_g2_affine_generator();
  EXPECT_EQ(c::base::native_cast(c_ret2),
            math::bn254::G2AffinePoint::Generator());
}

TEST_F(AffinePointTest, Random) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_random();
  EXPECT_NE(c::base::native_cast(c_ret), g1_a_);

  tachyon_bn254_g2_affine c_ret2 = tachyon_bn254_g2_affine_random();
  EXPECT_NE(c::base::native_cast(c_ret2), g2_a_);
}

TEST_F(AffinePointTest, Eq) {
  EXPECT_EQ(tachyon_bn254_g1_affine_eq(&g1_c_a_, &g1_c_b_), g1_a_ == g1_b_);
  EXPECT_EQ(tachyon_bn254_g2_affine_eq(&g2_c_a_, &g2_c_b_), g2_a_ == g2_b_);
}

TEST_F(AffinePointTest, Ne) {
  EXPECT_EQ(tachyon_bn254_g1_affine_ne(&g1_c_a_, &g1_c_b_), g1_a_ != g1_b_);
  EXPECT_EQ(tachyon_bn254_g2_affine_ne(&g2_c_a_, &g2_c_b_), g2_a_ != g2_b_);
}

TEST_F(AffinePointTest, Add) {
  tachyon_bn254_g1_jacobian c_ret =
      tachyon_bn254_g1_affine_add(&g1_c_a_, &g1_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_ + g1_b_);

  tachyon_bn254_g2_jacobian c_ret2 =
      tachyon_bn254_g2_affine_add(&g2_c_a_, &g2_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_ + g2_b_);
}

TEST_F(AffinePointTest, Sub) {
  tachyon_bn254_g1_jacobian c_ret =
      tachyon_bn254_g1_affine_sub(&g1_c_a_, &g1_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_ - g1_b_);

  tachyon_bn254_g2_jacobian c_ret2 =
      tachyon_bn254_g2_affine_sub(&g2_c_a_, &g2_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_ - g2_b_);
}

TEST_F(AffinePointTest, Neg) {
  tachyon_bn254_g1_affine c_ret = tachyon_bn254_g1_affine_neg(&g1_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret), -g1_a_);

  tachyon_bn254_g2_affine c_ret2 = tachyon_bn254_g2_affine_neg(&g2_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret2), -g2_a_);
}

TEST_F(AffinePointTest, Dbl) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_affine_dbl(&g1_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_.Double());

  tachyon_bn254_g2_jacobian c_ret2 = tachyon_bn254_g2_affine_dbl(&g2_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_.Double());
}

}  // namespace tachyon
