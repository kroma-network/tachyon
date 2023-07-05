#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test_config.h"

namespace tachyon {
namespace math {

namespace {

using Config = TestSwCurveConfig::Config;

class JacobianPointTest : public ::testing::Test {
 public:
  JacobianPointTest() {
    Fp7::Init();
    TestSwCurveConfig::Init();
  }
  JacobianPointTest(const JacobianPointTest&) = delete;
  JacobianPointTest& operator=(const JacobianPointTest&) = delete;
  ~JacobianPointTest() override = default;
};

}  // namespace

TEST_F(JacobianPointTest, IsZero) {
  EXPECT_TRUE(JacobianPoint<Config>(Fp7(1), Fp7(2), Fp7(0)).IsZero());
  EXPECT_FALSE(JacobianPoint<Config>(Fp7(1), Fp7(2), Fp7(1)).IsZero());
}

TEST_F(JacobianPointTest, EqualityOperators) {
  {
    // case 1) p.IsZero() && p2.IsZero()
    JacobianPoint<Config> p(Fp7(1), Fp7(2), Fp7(0));
    JacobianPoint<Config> p2(Fp7(3), Fp7(4), Fp7(0));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }

  {
    // case 2) !p.IsZero() && p2.IsZero()
    JacobianPoint<Config> p(Fp7(1), Fp7(2), Fp7(1));
    JacobianPoint<Config> p2(Fp7(3), Fp7(4), Fp7(0));
    EXPECT_TRUE(p != p2);
    EXPECT_TRUE(p2 != p);
  }

  {
    // other case
    JacobianPoint<Config> p(Fp7(1), Fp7(2), Fp7(3));
    JacobianPoint<Config> p2(Fp7(1), Fp7(2), Fp7(3));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }
}

TEST_F(JacobianPointTest, AdditiveGroupOperators) {
  JacobianPoint<Config> jp =
      JacobianPoint<Config>::CreateChecked(Fp7(5), Fp7(5), Fp7(1));
  JacobianPoint<Config> jp2 =
      JacobianPoint<Config>::CreateChecked(Fp7(3), Fp7(2), Fp7(1));
  JacobianPoint<Config> jp3 =
      JacobianPoint<Config>::CreateChecked(Fp7(3), Fp7(5), Fp7(1));
  JacobianPoint<Config> jp4 =
      JacobianPoint<Config>::CreateChecked(Fp7(6), Fp7(5), Fp7(1));
  AffinePoint<Config> ap = jp.ToAffine();
  AffinePoint<Config> ap2 = jp2.ToAffine();
  AffinePoint<Config> ap3 = jp3.ToAffine();
  AffinePoint<Config> ap4 = jp4.ToAffine();

  EXPECT_EQ(jp + jp2, jp3);
  EXPECT_EQ(jp - jp3, -jp2);
  EXPECT_EQ(jp + jp, jp4);
  EXPECT_EQ(jp - jp4, -jp);

  {
    JacobianPoint<Config> jp_tmp = jp;
    jp_tmp += jp2;
    EXPECT_EQ(jp_tmp, jp3);
    jp_tmp -= jp2;
    EXPECT_EQ(jp_tmp, jp);
  }

  EXPECT_EQ(jp + ap2, jp3);
  EXPECT_EQ(jp + ap, jp4);
  EXPECT_EQ(jp - jp3, -jp2);
  EXPECT_EQ(jp - jp4, -jp);
}

TEST_F(JacobianPointTest, ScalarMulOperator) {
  std::vector<AffinePoint<Config>> points;
  for (size_t i = 0; i < 7; ++i) {
    points.push_back((Fp7(i) * Config::Generator()).ToAffine());
  }

  EXPECT_THAT(points, ::testing::UnorderedElementsAreArray(
                          std::vector<AffinePoint<Config>>{
                              AffinePoint<Config>(Fp7(0), Fp7(0), true),
                              AffinePoint<Config>(Fp7(3), Fp7(2)),
                              AffinePoint<Config>(Fp7(5), Fp7(2)),
                              AffinePoint<Config>(Fp7(6), Fp7(2)),
                              AffinePoint<Config>(Fp7(3), Fp7(5)),
                              AffinePoint<Config>(Fp7(5), Fp7(5)),
                              AffinePoint<Config>(Fp7(6), Fp7(5))}));
}

TEST_F(JacobianPointTest, NegativeOperator) {
  JacobianPoint<Config> jp(Fp7(5), Fp7(5), Fp7(1));
  jp.NegativeInPlace();
  EXPECT_EQ(jp, JacobianPoint<Config>(Fp7(5), Fp7(-5), Fp7(1)));
}

TEST_F(JacobianPointTest, ToAffine) {
  EXPECT_EQ(JacobianPoint<Config>(Fp7(1), Fp7(2), Fp7(0)).ToAffine(),
            AffinePoint<Config>::Identity());
  EXPECT_EQ(JacobianPoint<Config>(Fp7(1), Fp7(2), Fp7(1)).ToAffine(),
            AffinePoint<Config>(Fp7(1), Fp7(2)));
  EXPECT_EQ(JacobianPoint<Config>(Fp7(1), Fp7(2), Fp7(3)).ToAffine(),
            AffinePoint<Config>(Fp7(4), Fp7(5)));
}

TEST_F(JacobianPointTest, MSM) {
  std::vector<JacobianPoint<Config>> bases = {
      {Fp7(5), Fp7(5), Fp7(1)},
      {Fp7(3), Fp7(2), Fp7(1)},
  };
  std::vector<Fp7> scalars = {
      Fp7(2),
      Fp7(3),
  };
  JacobianPoint<Config>::MSM(bases.begin(), bases.end(), scalars.begin(),
                             scalars.end());
}

}  // namespace math
}  // namespace tachyon
