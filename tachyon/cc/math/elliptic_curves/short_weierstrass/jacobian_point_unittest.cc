#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/point_conversions.h"
#include "tachyon/cc/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::cc::math {

namespace {

class JacobianPointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { tachyon::math::bn254::G1Curve::Init(); }

  void SetUp() override {
    a_ = tachyon::math::bn254::G1JacobianPoint::Random();
    b_ = tachyon::math::bn254::G1JacobianPoint::Random();

    cc_a_ = bn254::G1JacobianPoint(c::math::ToCJacobianPoint(a_));
    cc_b_ = bn254::G1JacobianPoint(c::math::ToCJacobianPoint(b_));
  }

 protected:
  tachyon::math::bn254::G1JacobianPoint a_;
  tachyon::math::bn254::G1JacobianPoint b_;
  bn254::G1JacobianPoint cc_a_;
  bn254::G1JacobianPoint cc_b_;
};

}  // namespace

TEST_F(JacobianPointTest, Zero) {
  bn254::G1JacobianPoint c_ret = bn254::G1JacobianPoint::Zero();
  EXPECT_TRUE(c::math::ToJacobianPoint(c_ret.ToCPoint()).IsZero());
}

TEST_F(JacobianPointTest, Generator) {
  bn254::G1JacobianPoint c_ret = bn254::G1JacobianPoint::Generator();
  EXPECT_EQ(c::math::ToJacobianPoint(c_ret.ToCPoint()),
            tachyon::math::bn254::G1JacobianPoint::Generator());
}

TEST_F(JacobianPointTest, Random) {
  bn254::G1JacobianPoint c_ret = bn254::G1JacobianPoint::Random();
  EXPECT_NE(c::math::ToJacobianPoint(c_ret.ToCPoint()), a_);
}

TEST_F(JacobianPointTest, Eq) { EXPECT_EQ(cc_a_ == cc_b_, a_ == b_); }

TEST_F(JacobianPointTest, Ne) { EXPECT_EQ(cc_a_ != cc_b_, a_ != b_); }

TEST_F(JacobianPointTest, Add) {
  bn254::G1JacobianPoint cc_ret = cc_a_ + cc_b_;
  EXPECT_EQ(c::math::ToJacobianPoint(cc_ret.ToCPoint()), a_ + b_);
  EXPECT_EQ(c::math::ToJacobianPoint((cc_a_ += cc_b_).ToCPoint()), a_ += b_);
}

TEST_F(JacobianPointTest, Sub) {
  bn254::G1JacobianPoint cc_ret = cc_a_ - cc_b_;
  EXPECT_EQ(c::math::ToJacobianPoint(cc_ret.ToCPoint()), a_ - b_);
  EXPECT_EQ(c::math::ToJacobianPoint((cc_a_ -= cc_b_).ToCPoint()), a_ -= b_);
}

TEST_F(JacobianPointTest, Neg) {
  bn254::G1JacobianPoint cc_ret = -cc_a_;
  EXPECT_EQ(c::math::ToJacobianPoint(cc_ret.ToCPoint()), -a_);
}

TEST_F(JacobianPointTest, Dbl) {
  bn254::G1JacobianPoint cc_ret = cc_a_.Double();
  EXPECT_EQ(c::math::ToJacobianPoint(cc_ret.ToCPoint()), a_.Double());
}

}  // namespace tachyon::cc::math
