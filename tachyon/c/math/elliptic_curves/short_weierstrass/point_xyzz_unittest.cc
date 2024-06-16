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

class PointXYZZTest : public testing::Test {
 public:
  void SetUp() override {
    tachyon_bn254_g1_init();
    tachyon_bn254_g2_init();

    g1_a_ = math::bn254::G1PointXYZZ::Random();
    g1_b_ = math::bn254::G1PointXYZZ::Random();
    g1_c_a_ = c::base::c_cast(g1_a_);
    g1_c_b_ = c::base::c_cast(g1_b_);

    g2_a_ = math::bn254::G2PointXYZZ::Random();
    g2_b_ = math::bn254::G2PointXYZZ::Random();
    g2_c_a_ = c::base::c_cast(g2_a_);
    g2_c_b_ = c::base::c_cast(g2_b_);
  }

 protected:
  math::bn254::G1PointXYZZ g1_a_;
  math::bn254::G1PointXYZZ g1_b_;
  tachyon_bn254_g1_xyzz g1_c_a_;
  tachyon_bn254_g1_xyzz g1_c_b_;

  math::bn254::G2PointXYZZ g2_a_;
  math::bn254::G2PointXYZZ g2_b_;
  tachyon_bn254_g2_xyzz g2_c_a_;
  tachyon_bn254_g2_xyzz g2_c_b_;
};

}  // namespace

TEST_F(PointXYZZTest, Zero) {
  tachyon_bn254_g1_xyzz c_ret = tachyon_bn254_g1_xyzz_zero();
  EXPECT_TRUE(c::base::native_cast(c_ret).IsZero());

  tachyon_bn254_g2_xyzz c_ret2 = tachyon_bn254_g2_xyzz_zero();
  EXPECT_TRUE(c::base::native_cast(c_ret2).IsZero());
}

TEST_F(PointXYZZTest, Generator) {
  tachyon_bn254_g1_xyzz c_ret = tachyon_bn254_g1_xyzz_generator();
  EXPECT_EQ(c::base::native_cast(c_ret), math::bn254::G1PointXYZZ::Generator());

  tachyon_bn254_g2_xyzz c_ret2 = tachyon_bn254_g2_xyzz_generator();
  EXPECT_EQ(c::base::native_cast(c_ret2),
            math::bn254::G2PointXYZZ::Generator());
}

TEST_F(PointXYZZTest, Random) {
  tachyon_bn254_g1_xyzz c_ret = tachyon_bn254_g1_xyzz_random();
  EXPECT_NE(c::base::native_cast(c_ret), g1_a_);

  tachyon_bn254_g2_xyzz c_ret2 = tachyon_bn254_g2_xyzz_random();
  EXPECT_NE(c::base::native_cast(c_ret2), g2_a_);
}

TEST_F(PointXYZZTest, Eq) {
  EXPECT_EQ(tachyon_bn254_g1_xyzz_eq(&g1_c_a_, &g1_c_b_), g1_a_ == g1_b_);
  EXPECT_EQ(tachyon_bn254_g2_xyzz_eq(&g2_c_a_, &g2_c_b_), g2_a_ == g2_b_);
}

TEST_F(PointXYZZTest, Ne) {
  EXPECT_EQ(tachyon_bn254_g1_xyzz_ne(&g1_c_a_, &g1_c_b_), g1_a_ != g1_b_);
  EXPECT_EQ(tachyon_bn254_g2_xyzz_ne(&g2_c_a_, &g2_c_b_), g2_a_ != g2_b_);
}

TEST_F(PointXYZZTest, Add) {
  tachyon_bn254_g1_xyzz c_ret = tachyon_bn254_g1_xyzz_add(&g1_c_a_, &g1_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_ + g1_b_);

  math::bn254::G1AffinePoint d = math::bn254::G1AffinePoint::Random();
  tachyon_bn254_g1_affine c_d = c::base::c_cast(d);
  c_ret = tachyon_bn254_g1_xyzz_add_mixed(&g1_c_a_, &c_d);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_ + d);

  tachyon_bn254_g2_xyzz c_ret2 = tachyon_bn254_g2_xyzz_add(&g2_c_a_, &g2_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_ + g2_b_);

  math::bn254::G2AffinePoint d2 = math::bn254::G2AffinePoint::Random();
  tachyon_bn254_g2_affine c_d2 = c::base::c_cast(d2);
  c_ret2 = tachyon_bn254_g2_xyzz_add_mixed(&g2_c_a_, &c_d2);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_ + d2);
}

TEST_F(PointXYZZTest, Sub) {
  tachyon_bn254_g1_xyzz c_ret = tachyon_bn254_g1_xyzz_sub(&g1_c_a_, &g1_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_ - g1_b_);

  math::bn254::G1AffinePoint d = math::bn254::G1AffinePoint::Random();
  tachyon_bn254_g1_affine c_d = c::base::c_cast(d);
  c_ret = tachyon_bn254_g1_xyzz_sub_mixed(&g1_c_a_, &c_d);
  EXPECT_EQ(c::base::native_cast(c_ret), g1_a_ - d);

  tachyon_bn254_g2_xyzz c_ret2 = tachyon_bn254_g2_xyzz_sub(&g2_c_a_, &g2_c_b_);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_ - g2_b_);

  math::bn254::G2AffinePoint d2 = math::bn254::G2AffinePoint::Random();
  tachyon_bn254_g2_affine c_d2 = c::base::c_cast(d2);
  c_ret2 = tachyon_bn254_g2_xyzz_sub_mixed(&g2_c_a_, &c_d2);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_ - d2);
}

TEST_F(PointXYZZTest, Neg) {
  tachyon_bn254_g1_xyzz c_ret = tachyon_bn254_g1_xyzz_neg(&g1_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret), -g1_a_);

  tachyon_bn254_g2_xyzz c_ret2 = tachyon_bn254_g2_xyzz_neg(&g2_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret2), -g2_a_);
}

TEST_F(PointXYZZTest, Dbl) {
  tachyon_bn254_g2_xyzz c_ret = tachyon_bn254_g2_xyzz_dbl(&g2_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret), g2_a_.Double());

  tachyon_bn254_g2_xyzz c_ret2 = tachyon_bn254_g2_xyzz_dbl(&g2_c_a_);
  EXPECT_EQ(c::base::native_cast(c_ret2), g2_a_.Double());
}

}  // namespace tachyon
