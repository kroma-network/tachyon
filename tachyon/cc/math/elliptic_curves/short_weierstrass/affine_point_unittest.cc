#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_prime_field_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/cc/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::cc::math {

namespace {

class AffinePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    tachyon::math::bn254::G1AffinePoint::Curve::Init();
  }

  void SetUp() override {
    a_ = tachyon::math::bn254::G1AffinePoint::Random();
    b_ = tachyon::math::bn254::G1AffinePoint::Random();

    cc_a_ = bn254::G1AffinePoint(ToCAffinePoint(a_));
    cc_b_ = bn254::G1AffinePoint(ToCAffinePoint(b_));
  }

 protected:
  tachyon::math::bn254::G1AffinePoint a_;
  tachyon::math::bn254::G1AffinePoint b_;
  bn254::G1AffinePoint cc_a_;
  bn254::G1AffinePoint cc_b_;
};

}  // namespace

TEST_F(AffinePointTest, Zero) {
  bn254::G1AffinePoint cc_ret = bn254::G1AffinePoint::Zero();
  EXPECT_TRUE(ToAffinePoint(cc_ret.ToCPoint()).IsZero());
}

TEST_F(AffinePointTest, Generator) {
  bn254::G1AffinePoint cc_ret = bn254::G1AffinePoint::Generator();
  EXPECT_EQ(ToAffinePoint(cc_ret.ToCPoint()),
            tachyon::math::bn254::G1AffinePoint::Generator());
}

TEST_F(AffinePointTest, Random) {
  bn254::G1AffinePoint cc_ret = bn254::G1AffinePoint::Random();
  EXPECT_NE(ToAffinePoint(cc_ret.ToCPoint()), a_);
}

TEST_F(AffinePointTest, Add) {
  bn254::G1JacobianPoint cc_ret = cc_a_ + cc_b_;
  EXPECT_EQ(ToJacobianPoint(cc_ret.ToCPoint()), a_ + b_);
}

TEST_F(AffinePointTest, Sub) {
  bn254::G1JacobianPoint cc_ret = cc_a_ - cc_b_;
  EXPECT_EQ(ToJacobianPoint(cc_ret.ToCPoint()), a_ - b_);
}

}  // namespace tachyon::cc::math
