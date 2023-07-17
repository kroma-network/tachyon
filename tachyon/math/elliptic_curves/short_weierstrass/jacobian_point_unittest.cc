#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test_config.h"

namespace tachyon {
namespace math {

namespace {

using Config = test::CurveConfig::Config;

class JacobianPointTest : public ::testing::Test {
 public:
  JacobianPointTest() {
    GF7Config::Init();
    test::CurveConfig::Init();
  }
  JacobianPointTest(const JacobianPointTest&) = delete;
  JacobianPointTest& operator=(const JacobianPointTest&) = delete;
  ~JacobianPointTest() override = default;
};

}  // namespace

TEST_F(JacobianPointTest, IsZero) {
  EXPECT_TRUE(JacobianPoint<Config>(GF7Gmp(1), GF7Gmp(2), GF7Gmp(0)).IsZero());
  EXPECT_FALSE(JacobianPoint<Config>(GF7Gmp(1), GF7Gmp(2), GF7Gmp(1)).IsZero());
}

TEST_F(JacobianPointTest, Random) {
  bool success = false;
  JacobianPoint<Config> r = JacobianPoint<Config>::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != JacobianPoint<Config>::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(JacobianPointTest, EqualityOperators) {
  {
    SCOPED_TRACE("p.IsZero() && p2.IsZero()");
    JacobianPoint<Config> p(GF7Gmp(1), GF7Gmp(2), GF7Gmp(0));
    JacobianPoint<Config> p2(GF7Gmp(3), GF7Gmp(4), GF7Gmp(0));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }

  {
    SCOPED_TRACE("!p.IsZero() && p2.IsZero()");
    JacobianPoint<Config> p(GF7Gmp(1), GF7Gmp(2), GF7Gmp(1));
    JacobianPoint<Config> p2(GF7Gmp(3), GF7Gmp(4), GF7Gmp(0));
    EXPECT_TRUE(p != p2);
    EXPECT_TRUE(p2 != p);
  }

  {
    SCOPED_TRACE("other");
    JacobianPoint<Config> p(GF7Gmp(1), GF7Gmp(2), GF7Gmp(3));
    JacobianPoint<Config> p2(GF7Gmp(1), GF7Gmp(2), GF7Gmp(3));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }
}

TEST_F(JacobianPointTest, AdditiveGroupOperators) {
  JacobianPoint<Config> jp =
      JacobianPoint<Config>::CreateChecked(GF7Gmp(5), GF7Gmp(5), GF7Gmp(1));
  JacobianPoint<Config> jp2 =
      JacobianPoint<Config>::CreateChecked(GF7Gmp(3), GF7Gmp(2), GF7Gmp(1));
  JacobianPoint<Config> jp3 =
      JacobianPoint<Config>::CreateChecked(GF7Gmp(3), GF7Gmp(5), GF7Gmp(1));
  JacobianPoint<Config> jp4 =
      JacobianPoint<Config>::CreateChecked(GF7Gmp(6), GF7Gmp(5), GF7Gmp(1));
  AffinePoint<Config> ap = jp.ToAffine();
  AffinePoint<Config> ap2 = jp2.ToAffine();

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
    points.push_back((GF7Gmp(i) * Config::Generator()).ToAffine());
  }

  EXPECT_THAT(points, ::testing::UnorderedElementsAreArray(
                          std::vector<AffinePoint<Config>>{
                              AffinePoint<Config>(GF7Gmp(0), GF7Gmp(0), true),
                              AffinePoint<Config>(GF7Gmp(3), GF7Gmp(2)),
                              AffinePoint<Config>(GF7Gmp(5), GF7Gmp(2)),
                              AffinePoint<Config>(GF7Gmp(6), GF7Gmp(2)),
                              AffinePoint<Config>(GF7Gmp(3), GF7Gmp(5)),
                              AffinePoint<Config>(GF7Gmp(5), GF7Gmp(5)),
                              AffinePoint<Config>(GF7Gmp(6), GF7Gmp(5))}));
}

TEST_F(JacobianPointTest, NegativeOperator) {
  JacobianPoint<Config> jp(GF7Gmp(5), GF7Gmp(5), GF7Gmp(1));
  jp.NegInPlace();
  EXPECT_EQ(jp, JacobianPoint<Config>(GF7Gmp(5), GF7Gmp(2), GF7Gmp(1)));
}

TEST_F(JacobianPointTest, ToAffine) {
  EXPECT_EQ(JacobianPoint<Config>(GF7Gmp(1), GF7Gmp(2), GF7Gmp(0)).ToAffine(),
            AffinePoint<Config>::Identity());
  EXPECT_EQ(JacobianPoint<Config>(GF7Gmp(1), GF7Gmp(2), GF7Gmp(1)).ToAffine(),
            AffinePoint<Config>(GF7Gmp(1), GF7Gmp(2)));
  EXPECT_EQ(JacobianPoint<Config>(GF7Gmp(1), GF7Gmp(2), GF7Gmp(3)).ToAffine(),
            AffinePoint<Config>(GF7Gmp(4), GF7Gmp(5)));
}

TEST_F(JacobianPointTest, MSM) {
  std::vector<JacobianPoint<Config>> bases = {
      {GF7Gmp(5), GF7Gmp(5), GF7Gmp(1)},
      {GF7Gmp(3), GF7Gmp(2), GF7Gmp(1)},
  };
  std::vector<GF7Gmp> scalars = {
      GF7Gmp(2),
      GF7Gmp(3),
  };
  JacobianPoint<Config> expected = JacobianPoint<Config>::Zero();
  for (size_t i = 0; i < bases.size(); ++i) {
    expected += bases[i].ScalarMul(scalars[i].ToBigInt());
  }
  EXPECT_EQ(JacobianPoint<Config>::MSM(bases, scalars), expected);
}

}  // namespace math
}  // namespace tachyon
