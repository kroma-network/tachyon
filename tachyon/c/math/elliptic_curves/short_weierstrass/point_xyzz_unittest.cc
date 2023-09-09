#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_prime_field_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::c::math {

namespace {

class PointXYZZTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    tachyon::math::bn254::G1PointXYZZ::Curve::Init();
  }

  void SetUp() override {
    a_ = tachyon::math::bn254::G1PointXYZZ::Random();
    b_ = tachyon::math::bn254::G1PointXYZZ::Random();

    c_a_ = cc::math::ToCPointXYZZ(a_);
    c_b_ = cc::math::ToCPointXYZZ(b_);
  }

 protected:
  tachyon::math::bn254::G1PointXYZZ a_;
  tachyon::math::bn254::G1PointXYZZ b_;
  tachyon_bn254_g1_xyzz c_a_;
  tachyon_bn254_g1_xyzz c_b_;
};

}  // namespace

TEST_F(PointXYZZTest, Zero) {
  tachyon_bn254_g1_xyzz c_ret = tachyon_bn254_g1_xyzz_zero();
  EXPECT_TRUE(cc::math::ToPointXYZZ(c_ret).IsZero());
}

TEST_F(PointXYZZTest, Generator) {
  tachyon_bn254_g1_xyzz c_ret = tachyon_bn254_g1_xyzz_generator();
  EXPECT_EQ(cc::math::ToPointXYZZ(c_ret),
            tachyon::math::bn254::G1PointXYZZ::Generator());
}

TEST_F(PointXYZZTest, Random) {
  tachyon_bn254_g1_xyzz c_ret = tachyon_bn254_g1_xyzz_random();
  EXPECT_NE(cc::math::ToPointXYZZ(c_ret), a_);
}

TEST_F(PointXYZZTest, Eq) {
  EXPECT_EQ(tachyon_bn254_g1_xyzz_eq(&c_a_, &c_b_), a_ == b_);
}

TEST_F(PointXYZZTest, Ne) {
  EXPECT_EQ(tachyon_bn254_g1_xyzz_ne(&c_a_, &c_b_), a_ != b_);
}

}  // namespace tachyon::c::math
