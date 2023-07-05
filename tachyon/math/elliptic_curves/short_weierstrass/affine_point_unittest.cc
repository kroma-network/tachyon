#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test_config.h"

namespace tachyon {
namespace math {

namespace {

using Config = TestSwCurveConfig::Config;

class AffinePointTest : public ::testing::Test {
 public:
  AffinePointTest() {
    Fp7::Init();
    TestSwCurveConfig::Init();
  }
  AffinePointTest(const AffinePointTest&) = delete;
  AffinePointTest& operator=(const AffinePointTest&) = delete;
  ~AffinePointTest() override = default;
};

}  // namespace

TEST_F(AffinePointTest, Zero) {
  EXPECT_TRUE(AffinePoint<Config>::Zero().infinity());
}

TEST_F(AffinePointTest, EqualityOperator) {
  AffinePoint<Config> p(Fp7(1), Fp7(2));
  AffinePoint<Config> p2(Fp7(3), Fp7(4));
  EXPECT_TRUE(p == p);
  EXPECT_TRUE(p != p2);
}

TEST_F(AffinePointTest, AdditiveGroupOperators) {
  AffinePoint<Config> ap = AffinePoint<Config>::CreateChecked(Fp7(5), Fp7(5));
  AffinePoint<Config> ap2 = AffinePoint<Config>::CreateChecked(Fp7(3), Fp7(2));
  AffinePoint<Config> ap3 = AffinePoint<Config>::CreateChecked(Fp7(3), Fp7(5));
  AffinePoint<Config> ap4 = AffinePoint<Config>::CreateChecked(Fp7(6), Fp7(5));
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
  AffinePoint<Config> p(Fp7(3), Fp7(2));
  EXPECT_EQ(p.ToJacobian(), JacobianPoint<Config>(Fp7(3), Fp7(2), Fp7(1)));
}

TEST_F(AffinePointTest, IsOnCurve) {
  AffinePoint<Config> invalid_point(Fp7(1), Fp7(2));
  EXPECT_FALSE(AffinePoint<Config>::IsOnCurve(invalid_point));
  AffinePoint<Config> valid_point(Fp7(3), Fp7(2));
  EXPECT_TRUE(AffinePoint<Config>::IsOnCurve(valid_point));
  valid_point = AffinePoint<Config>(Fp7(3), Fp7(5));
  EXPECT_TRUE(AffinePoint<Config>::IsOnCurve(valid_point));
}

TEST_F(AffinePointTest, MSM) {
  std::vector<AffinePoint<Config>> bases = {
      {Fp7(5), Fp7(5)},
      {Fp7(3), Fp7(2)},
  };
  std::vector<Fp7> scalars = {
      Fp7(2),
      Fp7(3),
  };
  AffinePoint<Config>::MSM(bases.begin(), bases.end(), scalars.begin(),
                           scalars.end());
}

}  // namespace math
}  // namespace tachyon
