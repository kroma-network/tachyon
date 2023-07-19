#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test_config.h"

namespace tachyon {
namespace math {

namespace {

using Config = test::CurveConfig::Config;

class AffinePointTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    GF7Config::Init();
    test::CurveConfig::Init();
  }
};

}  // namespace

TEST_F(AffinePointTest, Zero) {
  EXPECT_TRUE(AffinePoint<Config>::Zero().infinity());
}

TEST_F(AffinePointTest, Random) {
  bool success = false;
  AffinePoint<Config> r = AffinePoint<Config>::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != AffinePoint<Config>::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(AffinePointTest, EqualityOperator) {
  AffinePoint<Config> p(GF7(1), GF7(2));
  AffinePoint<Config> p2(GF7(3), GF7(4));
  EXPECT_TRUE(p == p);
  EXPECT_TRUE(p != p2);
}

TEST_F(AffinePointTest, AdditiveGroupOperators) {
  AffinePoint<Config> ap = AffinePoint<Config>::CreateChecked(GF7(5), GF7(5));
  AffinePoint<Config> ap2 = AffinePoint<Config>::CreateChecked(GF7(3), GF7(2));
  AffinePoint<Config> ap3 = AffinePoint<Config>::CreateChecked(GF7(3), GF7(5));
  AffinePoint<Config> ap4 = AffinePoint<Config>::CreateChecked(GF7(6), GF7(5));
  JacobianPoint<Config> jp = ap.ToJacobian();
  JacobianPoint<Config> jp2 = ap2.ToJacobian();
  JacobianPoint<Config> jp3 = ap3.ToJacobian();
  JacobianPoint<Config> jp4 = ap4.ToJacobian();

  EXPECT_EQ(ap + ap2, jp3);
  EXPECT_EQ(ap + ap, jp4);
  EXPECT_EQ(ap3 - ap2, jp);
  EXPECT_EQ(ap4 - ap, jp);

  EXPECT_EQ(ap + jp2, jp3);
  EXPECT_EQ(ap + jp, jp4);
  EXPECT_EQ(ap - jp3, -jp2);
  EXPECT_EQ(ap - jp4, -jp);
}

TEST_F(AffinePointTest, ToJacobian) {
  EXPECT_EQ(AffinePoint<Config>::Identity().ToJacobian(),
            JacobianPoint<Config>::Zero());
  AffinePoint<Config> p(GF7(3), GF7(2));
  EXPECT_EQ(p.ToJacobian(), JacobianPoint<Config>(GF7(3), GF7(2), GF7(1)));
}

TEST_F(AffinePointTest, IsOnCurve) {
  AffinePoint<Config> invalid_point(GF7(1), GF7(2));
  EXPECT_FALSE(AffinePoint<Config>::IsOnCurve(invalid_point));
  AffinePoint<Config> valid_point(GF7(3), GF7(2));
  EXPECT_TRUE(AffinePoint<Config>::IsOnCurve(valid_point));
  valid_point = AffinePoint<Config>(GF7(3), GF7(5));
  EXPECT_TRUE(AffinePoint<Config>::IsOnCurve(valid_point));
}

TEST_F(AffinePointTest, MSM) {
  std::vector<AffinePoint<Config>> bases = {
      {GF7(5), GF7(5)},
      {GF7(3), GF7(2)},
  };
  std::vector<GF7> scalars = {
      GF7(2),
      GF7(3),
  };
  JacobianPoint<Config> expected = JacobianPoint<Config>::Zero();
  for (size_t i = 0; i < bases.size(); ++i) {
    expected += bases[i].ToJacobian().ScalarMul(scalars[i].ToBigInt());
  }
  EXPECT_EQ(AffinePoint<Config>::MSM(bases, scalars), expected);
}

}  // namespace math
}  // namespace tachyon
