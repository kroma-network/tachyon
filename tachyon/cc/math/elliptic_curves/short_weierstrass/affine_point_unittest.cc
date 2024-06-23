#include "gtest/gtest.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fq_type_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_traits.h"
#include "tachyon/c/math/elliptic_curves/bn/bn254/g1_point_type_traits.h"
#include "tachyon/cc/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::cc::math {

namespace {

class AffinePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { tachyon::math::bn254::G1Curve::Init(); }

  void SetUp() override {
    a_ = tachyon::math::bn254::G1AffinePoint::Random();
    b_ = tachyon::math::bn254::G1AffinePoint::Random();

    cc_a_ = bn254::G1AffinePoint(c::base::c_cast(a_));
    cc_b_ = bn254::G1AffinePoint(c::base::c_cast(b_));
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
  EXPECT_TRUE(c::base::native_cast(cc_ret.ToCPoint()).IsZero());
}

TEST_F(AffinePointTest, Generator) {
  bn254::G1AffinePoint cc_ret = bn254::G1AffinePoint::Generator();
  EXPECT_EQ(c::base::native_cast(cc_ret.ToCPoint()),
            tachyon::math::bn254::G1AffinePoint::Generator());
}

TEST_F(AffinePointTest, Random) {
  bn254::G1AffinePoint cc_ret = bn254::G1AffinePoint::Random();
  EXPECT_NE(c::base::native_cast(cc_ret.ToCPoint()), a_);
}

TEST_F(AffinePointTest, Eq) { EXPECT_EQ(cc_a_ == cc_b_, a_ == b_); }

TEST_F(AffinePointTest, Ne) { EXPECT_EQ(cc_a_ != cc_b_, a_ != b_); }

TEST_F(AffinePointTest, Add) {
  bn254::G1JacobianPoint cc_ret = cc_a_ + cc_b_;
  EXPECT_EQ(c::base::native_cast(cc_ret.ToCPoint()), a_ + b_);
}

TEST_F(AffinePointTest, Sub) {
  bn254::G1JacobianPoint cc_ret = cc_a_ - cc_b_;
  EXPECT_EQ(c::base::native_cast(cc_ret.ToCPoint()), a_ - b_);
}

TEST_F(AffinePointTest, Neg) {
  bn254::G1AffinePoint cc_ret = -cc_a_;
  EXPECT_EQ(c::base::native_cast(cc_ret.ToCPoint()), -a_);
}

TEST_F(AffinePointTest, Dbl) {
  bn254::G1JacobianPoint cc_ret = cc_a_.Double();
  EXPECT_EQ(c::base::native_cast(cc_ret.ToCPoint()), a_.Double());
}

}  // namespace tachyon::cc::math
