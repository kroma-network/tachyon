#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g2_point_type_traits.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"

namespace tachyon {

namespace {

class JacobianPointTest : public testing::Test {
 public:
  void SetUp() override {
    tachyon_bn254_g1_init();
    tachyon_bn254_g2_init();

    g1_a_ = math::bn254::G1JacobianPoint::Random();
    g1_b_ = math::bn254::G1JacobianPoint::Random();
    g1_c_a_ = c::base::c_cast(g1_a_);
    g1_c_b_ = c::base::c_cast(g1_b_);

    g2_a_ = math::bn254::G2JacobianPoint::Random();
    g2_b_ = math::bn254::G2JacobianPoint::Random();
    g2_c_a_ = c::base::c_cast(g2_a_);
    g2_c_b_ = c::base::c_cast(g2_b_);
  }

 protected:
  math::bn254::G1JacobianPoint g1_a_;
  math::bn254::G1JacobianPoint g1_b_;
  tachyon_bn254_g1_jacobian g1_c_a_;
  tachyon_bn254_g1_jacobian g1_c_b_;

  math::bn254::G2JacobianPoint g2_a_;
  math::bn254::G2JacobianPoint g2_b_;
  tachyon_bn254_g2_jacobian g2_c_a_;
  tachyon_bn254_g2_jacobian g2_c_b_;
};

}  // namespace

TEST_F(JacobianPointTest, Zero) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_jacobian_zero();
  EXPECT_TRUE(c::base::native_cast(c_ret).IsZero());

  tachyon_bn254_g2_jacobian c_ret2 = tachyon_bn254_g2_jacobian_zero();
  EXPECT_TRUE(c::base::native_cast(c_ret2).IsZero());
}

TEST_F(JacobianPointTest, Generator) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_jacobian_generator();
  EXPECT_EQ(c::base::native_cast(c_ret),
            math::bn254::G1JacobianPoint::Generator());

  tachyon_bn254_g2_jacobian c_ret2 = tachyon_bn254_g2_jacobian_generator();
  EXPECT_EQ(c::base::native_cast(c_ret2),
            math::bn254::G2JacobianPoint::Generator());
}

TEST_F(JacobianPointTest, Random) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_jacobian_random();
  EXPECT_NE(c::base::native_cast(c_ret), g1_a_);

  tachyon_bn254_g2_jacobian c_ret2 = tachyon_bn254_g2_jacobian_random();
  EXPECT_NE(c::base::native_cast(c_ret2), g2_a_);
}

TEST_F(JacobianPointTest, Eq) {
  EXPECT_EQ(tachyon_bn254_g1_jacobian_eq(&g1_c_a_, &g1_c_b_), g1_a_ == g1_b_);
  EXPECT_EQ(tachyon_bn254_g2_jacobian_eq(&g2_c_a_, &g2_c_b_), g2_a_ == g2_b_);
}

TEST_F(JacobianPointTest, Ne) {
  EXPECT_EQ(tachyon_bn254_g1_jacobian_ne(&g1_c_a_, &g1_c_b_), g1_a_ != g1_b_);
  EXPECT_EQ(tachyon_bn254_g2_jacobian_ne(&g2_c_a_, &g2_c_b_), g2_a_ != g2_b_);
}

TEST_F(JacobianPointTest, Add) {
  tachyon_bn254_g1_jacobian c_ret =
      tachyon_bn254_g1_jacobian_add(&g1_c_a_, &g1_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_ + g1_b_);

  math::bn254::G1AffinePoint d = math::bn254::G1AffinePoint::Random();
  tachyon_bn254_g1_affine c_d = c::base::c_cast(d);
  c_ret = tachyon_bn254_g1_jacobian_add_mixed(&g1_c_a_, &c_d);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_ + d);

  tachyon_bn254_g2_jacobian c_ret2 =
      tachyon_bn254_g2_jacobian_add(&g2_c_a_, &g2_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_ + g2_b_);

  math::bn254::G2AffinePoint d2 = math::bn254::G2AffinePoint::Random();
  tachyon_bn254_g2_affine c_d2 = c::base::c_cast(d2);
  c_ret2 = tachyon_bn254_g2_jacobian_add_mixed(&g2_c_a_, &c_d2);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_ + d2);
}

TEST_F(JacobianPointTest, Sub) {
  tachyon_bn254_g1_jacobian c_ret =
      tachyon_bn254_g1_jacobian_sub(&g1_c_a_, &g1_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_ - g1_b_);

  math::bn254::G1AffinePoint d = math::bn254::G1AffinePoint::Random();
  tachyon_bn254_g1_affine c_d = c::base::c_cast(d);
  c_ret = tachyon_bn254_g1_jacobian_sub_mixed(&g1_c_a_, &c_d);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_ - d);

  tachyon_bn254_g2_jacobian c_ret2 =
      tachyon_bn254_g2_jacobian_sub(&g2_c_a_, &g2_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_ - g2_b_);

  math::bn254::G2AffinePoint d2 = math::bn254::G2AffinePoint::Random();
  tachyon_bn254_g2_affine c_d2 = c::base::c_cast(d2);
  c_ret2 = tachyon_bn254_g2_jacobian_sub_mixed(&g2_c_a_, &c_d2);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_ - d2);
}

TEST_F(JacobianPointTest, Neg) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_jacobian_neg(&g1_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret), -g1_a_);

  tachyon_bn254_g2_jacobian c_ret2 = tachyon_bn254_g2_jacobian_neg(&g2_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret2), -g2_a_);
}

TEST_F(JacobianPointTest, Dbl) {
  tachyon_bn254_g1_jacobian c_ret = tachyon_bn254_g1_jacobian_dbl(&g1_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_.Double());

  tachyon_bn254_g2_jacobian c_ret2 = tachyon_bn254_g2_jacobian_dbl(&g2_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_.Double());
}

}  // namespace tachyon
