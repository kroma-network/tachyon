#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/sw_curve_config.h"

namespace tachyon::math {

namespace {

class JacobianPointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { test::G1Curve::Init(); }
};

}  // namespace

TEST_F(JacobianPointTest, IsZero) {
  EXPECT_TRUE(test::JacobianPoint::Zero().IsZero());
  EXPECT_FALSE(test::JacobianPoint(GF7(1), GF7(2), GF7(1)).IsZero());
}

TEST_F(JacobianPointTest, Generator) {
  EXPECT_EQ(test::JacobianPoint::Generator(),
            test::JacobianPoint(
                test::JacobianPoint::Curve::Config::kGenerator.x,
                test::JacobianPoint::Curve::Config::kGenerator.y, GF7::One()));
}

TEST_F(JacobianPointTest, Montgomery) {
  test::JacobianPoint r = test::JacobianPoint::Random();
  EXPECT_EQ(r, test::JacobianPoint::FromMontgomery(r.ToMontgomery()));
}

TEST_F(JacobianPointTest, Random) {
  bool success = false;
  test::JacobianPoint r = test::JacobianPoint::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != test::JacobianPoint::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(JacobianPointTest, EqualityOperators) {
  {
    SCOPED_TRACE("p.IsZero() && p2.IsZero()");
    test::JacobianPoint p(GF7(1), GF7(2), GF7(0));
    test::JacobianPoint p2(GF7(3), GF7(4), GF7(0));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }

  {
    SCOPED_TRACE("!p.IsZero() && p2.IsZero()");
    test::JacobianPoint p(GF7(1), GF7(2), GF7(1));
    test::JacobianPoint p2(GF7(3), GF7(4), GF7(0));
    EXPECT_TRUE(p != p2);
    EXPECT_TRUE(p2 != p);
  }

  {
    SCOPED_TRACE("other");
    test::JacobianPoint p(GF7(1), GF7(2), GF7(3));
    test::JacobianPoint p2(GF7(1), GF7(2), GF7(3));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }
}

TEST_F(JacobianPointTest, AdditiveGroupOperators) {
  test::JacobianPoint jp =
      test::JacobianPoint::CreateChecked(GF7(5), GF7(5), GF7(1));
  test::JacobianPoint jp2 =
      test::JacobianPoint::CreateChecked(GF7(3), GF7(2), GF7(1));
  test::JacobianPoint jp3 =
      test::JacobianPoint::CreateChecked(GF7(3), GF7(5), GF7(1));
  test::JacobianPoint jp4 =
      test::JacobianPoint::CreateChecked(GF7(6), GF7(5), GF7(1));
  test::AffinePoint ap = jp.ToAffine();
  test::AffinePoint ap2 = jp2.ToAffine();

  EXPECT_EQ(jp + jp2, jp3);
  EXPECT_EQ(jp - jp3, -jp2);
  EXPECT_EQ(jp + jp, jp4);
  EXPECT_EQ(jp - jp4, -jp);

  {
    test::JacobianPoint jp_tmp = jp;
    jp_tmp += jp2;
    EXPECT_EQ(jp_tmp, jp3);
    jp_tmp -= jp2;
    EXPECT_EQ(jp_tmp, jp);
  }

  EXPECT_EQ(jp + ap2, jp3);
  EXPECT_EQ(jp + ap, jp4);
  EXPECT_EQ(jp - jp3, -jp2);
  EXPECT_EQ(jp - jp4, -jp);

  EXPECT_EQ(jp.Double(), jp4);
  {
    test::JacobianPoint jp_tmp = jp;
    jp_tmp.DoubleInPlace();
    EXPECT_EQ(jp_tmp, jp4);
  }

  EXPECT_EQ(-jp, test::JacobianPoint(GF7(5), GF7(2), GF7(1)));
  {
    test::JacobianPoint jp_tmp = jp;
    jp_tmp.NegInPlace();
    EXPECT_EQ(jp_tmp, test::JacobianPoint(GF7(5), GF7(2), GF7(1)));
  }

  EXPECT_EQ(jp * GF7(2), jp4);
  EXPECT_EQ(GF7(2) * jp, jp4);
  EXPECT_EQ(jp *= GF7(2), jp4);
}

TEST_F(JacobianPointTest, ScalarMulOperator) {
  std::vector<test::AffinePoint> points;
  for (size_t i = 0; i < 7; ++i) {
    points.push_back((GF7(i) * test::JacobianPoint::Generator()).ToAffine());
  }

  EXPECT_THAT(
      points,
      testing::UnorderedElementsAreArray(std::vector<test::AffinePoint>{
          test::AffinePoint(GF7(0), GF7(0), true),
          test::AffinePoint(GF7(3), GF7(2)), test::AffinePoint(GF7(5), GF7(2)),
          test::AffinePoint(GF7(6), GF7(2)), test::AffinePoint(GF7(3), GF7(5)),
          test::AffinePoint(GF7(5), GF7(5)),
          test::AffinePoint(GF7(6), GF7(5))}));
}

TEST_F(JacobianPointTest, ToAffine) {
  EXPECT_EQ(test::JacobianPoint(GF7(1), GF7(2), GF7(0)).ToAffine(),
            test::AffinePoint::Zero());
  EXPECT_EQ(test::JacobianPoint(GF7(1), GF7(2), GF7(1)).ToAffine(),
            test::AffinePoint(GF7(1), GF7(2)));
  EXPECT_EQ(test::JacobianPoint(GF7(1), GF7(2), GF7(3)).ToAffine(),
            test::AffinePoint(GF7(4), GF7(5)));
}

TEST_F(JacobianPointTest, ToProjective) {
  EXPECT_EQ(test::JacobianPoint(GF7(1), GF7(2), GF7(0)).ToProjective(),
            test::ProjectivePoint::Zero());
  EXPECT_EQ(test::JacobianPoint(GF7(1), GF7(2), GF7(3)).ToProjective(),
            test::ProjectivePoint(GF7(3), GF7(2), GF7(6)));
}

TEST_F(JacobianPointTest, ToXYZZ) {
  EXPECT_EQ(test::JacobianPoint(GF7(1), GF7(2), GF7(0)).ToXYZZ(),
            test::PointXYZZ::Zero());
  EXPECT_EQ(test::JacobianPoint(GF7(1), GF7(2), GF7(3)).ToXYZZ(),
            test::PointXYZZ(GF7(1), GF7(2), GF7(2), GF7(6)));
}

TEST_F(JacobianPointTest, BatchNormalize) {
  std::vector<test::JacobianPoint> jacobian_points = {
      test::JacobianPoint(GF7(1), GF7(2), GF7(0)),
      test::JacobianPoint(GF7(1), GF7(2), GF7(1)),
      test::JacobianPoint(GF7(1), GF7(2), GF7(3))};

  std::vector<test::AffinePoint> affine_points;
  affine_points.resize(2);
  ASSERT_FALSE(
      test::JacobianPoint::BatchNormalize(jacobian_points, &affine_points));

  affine_points.resize(3);
  ASSERT_TRUE(
      test::JacobianPoint::BatchNormalize(jacobian_points, &affine_points));

  std::vector<test::AffinePoint> expected_affine_points = {
      test::AffinePoint::Zero(), test::AffinePoint(GF7(1), GF7(2)),
      test::AffinePoint(GF7(4), GF7(5))};
  EXPECT_EQ(affine_points, expected_affine_points);
}

TEST_F(JacobianPointTest, IsOnCurve) {
  test::JacobianPoint invalid_point(GF7(1), GF7(2), GF7(1));
  EXPECT_FALSE(invalid_point.IsOnCurve());
  test::JacobianPoint valid_point(GF7(3), GF7(2), GF7(1));
  EXPECT_TRUE(valid_point.IsOnCurve());
  valid_point = test::JacobianPoint(GF7(3), GF7(5), GF7(1));
  EXPECT_TRUE(valid_point.IsOnCurve());
}

TEST_F(JacobianPointTest, CreateFromX) {
  {
    std::optional<test::JacobianPoint> p =
        test::JacobianPoint::CreateFromX(GF7(3), /*pick_odd=*/true);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->y(), GF7(5));
  }
  {
    std::optional<test::JacobianPoint> p =
        test::JacobianPoint::CreateFromX(GF7(3), /*pick_odd=*/false);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->y(), GF7(2));
  }
  {
    std::optional<test::JacobianPoint> p =
        test::JacobianPoint::CreateFromX(GF7(1), /*pick_odd=*/false);
    ASSERT_FALSE(p.has_value());
  }
}

}  // namespace tachyon::math
