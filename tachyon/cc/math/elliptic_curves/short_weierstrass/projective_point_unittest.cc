#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_prime_field_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/cc/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/cc/math/elliptic_curves/point_conversions.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::cc::math {

namespace {

class ProjectivePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { tachyon::math::bn254::G1Curve::Init(); }

  void SetUp() override {
    a_ = tachyon::math::bn254::G1ProjectivePoint::Random();
    b_ = tachyon::math::bn254::G1ProjectivePoint::Random();

    cc_a_ = bn254::G1ProjectivePoint(ToCProjectivePoint(a_));
    cc_b_ = bn254::G1ProjectivePoint(ToCProjectivePoint(b_));
  }

 protected:
  tachyon::math::bn254::G1ProjectivePoint a_;
  tachyon::math::bn254::G1ProjectivePoint b_;
  bn254::G1ProjectivePoint cc_a_;
  bn254::G1ProjectivePoint cc_b_;
};

}  // namespace

TEST_F(ProjectivePointTest, Zero) {
  bn254::G1ProjectivePoint c_ret = bn254::G1ProjectivePoint::Zero();
  EXPECT_TRUE(ToProjectivePoint(c_ret.ToCPoint()).IsZero());
}

TEST_F(ProjectivePointTest, Generator) {
  bn254::G1ProjectivePoint c_ret = bn254::G1ProjectivePoint::Generator();
  EXPECT_EQ(ToProjectivePoint(c_ret.ToCPoint()),
            tachyon::math::bn254::G1ProjectivePoint::Generator());
}

TEST_F(ProjectivePointTest, Random) {
  bn254::G1ProjectivePoint c_ret = bn254::G1ProjectivePoint::Random();
  EXPECT_NE(ToProjectivePoint(c_ret.ToCPoint()), a_);
}

TEST_F(ProjectivePointTest, Eq) { EXPECT_EQ(cc_a_ == cc_b_, a_ == b_); }

TEST_F(ProjectivePointTest, Ne) { EXPECT_EQ(cc_a_ != cc_b_, a_ != b_); }

TEST_F(ProjectivePointTest, Add) {
  bn254::G1ProjectivePoint cc_ret = cc_a_ + cc_b_;
  EXPECT_EQ(ToProjectivePoint(cc_ret.ToCPoint()), a_ + b_);
  EXPECT_EQ(ToProjectivePoint((cc_a_ += cc_b_).ToCPoint()), a_ += b_);
}

TEST_F(ProjectivePointTest, Sub) {
  bn254::G1ProjectivePoint cc_ret = cc_a_ - cc_b_;
  EXPECT_EQ(ToProjectivePoint(cc_ret.ToCPoint()), a_ - b_);
  EXPECT_EQ(ToProjectivePoint((cc_a_ -= cc_b_).ToCPoint()), a_ -= b_);
}

TEST_F(ProjectivePointTest, Neg) {
  bn254::G1ProjectivePoint cc_ret = -cc_a_;
  EXPECT_EQ(ToProjectivePoint(cc_ret.ToCPoint()), -a_);
}

TEST_F(ProjectivePointTest, Dbl) {
  bn254::G1ProjectivePoint cc_ret = cc_a_.Double();
  EXPECT_EQ(ToProjectivePoint(cc_ret.ToCPoint()), a_.Double());
}

}  // namespace tachyon::cc::math
