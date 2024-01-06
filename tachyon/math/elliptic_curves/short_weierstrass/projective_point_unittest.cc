#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/sw_curve_config.h"

namespace tachyon::math {

namespace {

class ProjectivePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { test::G1Curve::Init(); }
};

}  // namespace

TEST_F(ProjectivePointTest, IsZero) {
  EXPECT_TRUE(test::ProjectivePoint::Zero().IsZero());
  EXPECT_FALSE(test::ProjectivePoint(GF7(1), GF7(2), GF7(1)).IsZero());
}

TEST_F(ProjectivePointTest, Generator) {
  EXPECT_EQ(
      test::ProjectivePoint::Generator(),
      test::ProjectivePoint(test::ProjectivePoint::Curve::Config::kGenerator.x,
                            test::ProjectivePoint::Curve::Config::kGenerator.y,
                            GF7::One()));
}

TEST_F(ProjectivePointTest, Montgomery) {
  test::ProjectivePoint r = test::ProjectivePoint::Random();
  EXPECT_EQ(r, test::ProjectivePoint::FromMontgomery(r.ToMontgomery()));
}

TEST_F(ProjectivePointTest, Random) {
  bool success = false;
  test::ProjectivePoint r = test::ProjectivePoint::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != test::ProjectivePoint::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(ProjectivePointTest, EqualityOperators) {
  {
    SCOPED_TRACE("p.IsZero() && p2.IsZero()");
    test::ProjectivePoint p(GF7(1), GF7(2), GF7(0));
    test::ProjectivePoint p2(GF7(3), GF7(4), GF7(0));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }

  {
    SCOPED_TRACE("!p.IsZero() && p2.IsZero()");
    test::ProjectivePoint p(GF7(1), GF7(2), GF7(1));
    test::ProjectivePoint p2(GF7(3), GF7(4), GF7(0));
    EXPECT_TRUE(p != p2);
    EXPECT_TRUE(p2 != p);
  }

  {
    SCOPED_TRACE("other");
    test::ProjectivePoint p(GF7(1), GF7(2), GF7(3));
    test::ProjectivePoint p2(GF7(1), GF7(2), GF7(3));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }
}

TEST_F(ProjectivePointTest, AdditiveGroupOperators) {
  test::ProjectivePoint pp =
      test::ProjectivePoint::CreateChecked(GF7(5), GF7(5), GF7(1));
  test::ProjectivePoint pp2 =
      test::ProjectivePoint::CreateChecked(GF7(3), GF7(2), GF7(1));
  test::ProjectivePoint pp3 =
      test::ProjectivePoint::CreateChecked(GF7(3), GF7(5), GF7(1));
  test::ProjectivePoint pp4 =
      test::ProjectivePoint::CreateChecked(GF7(6), GF7(5), GF7(1));
  test::AffinePoint ap = pp.ToAffine();
  test::AffinePoint ap2 = pp2.ToAffine();

  EXPECT_EQ(pp + pp2, pp3);
  EXPECT_EQ(pp - pp3, -pp2);
  EXPECT_EQ(pp + pp, pp4);
  EXPECT_EQ(pp - pp4, -pp);

  {
    test::ProjectivePoint pp_tmp = pp;
    pp_tmp += pp2;
    EXPECT_EQ(pp_tmp, pp3);
    pp_tmp -= pp2;
    EXPECT_EQ(pp_tmp, pp);
  }

  EXPECT_EQ(pp + ap2, pp3);
  EXPECT_EQ(pp + ap, pp4);
  EXPECT_EQ(pp - pp3, -pp2);
  EXPECT_EQ(pp - pp4, -pp);

  EXPECT_EQ(pp.Double(), pp4);
  {
    test::ProjectivePoint pp_tmp = pp;
    pp_tmp.DoubleInPlace();
    EXPECT_EQ(pp_tmp, pp4);
  }

  EXPECT_EQ(-pp, test::ProjectivePoint(GF7(5), GF7(2), GF7(1)));
  {
    test::ProjectivePoint pp_tmp = pp;
    pp_tmp.NegInPlace();
    EXPECT_EQ(pp_tmp, test::ProjectivePoint(GF7(5), GF7(2), GF7(1)));
  }

  EXPECT_EQ(pp * GF7(2), pp4);
  EXPECT_EQ(GF7(2) * pp, pp4);
  EXPECT_EQ(pp *= GF7(2), pp4);
}

TEST_F(ProjectivePointTest, ScalarMulOperator) {
  std::vector<test::AffinePoint> points;
  for (size_t i = 0; i < 7; ++i) {
    points.push_back((GF7(i) * test::ProjectivePoint::Generator()).ToAffine());
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

TEST_F(ProjectivePointTest, ToAffine) {
  EXPECT_EQ(test::ProjectivePoint(GF7(1), GF7(2), GF7(0)).ToAffine(),
            test::AffinePoint::Zero());
  EXPECT_EQ(test::ProjectivePoint(GF7(1), GF7(2), GF7(1)).ToAffine(),
            test::AffinePoint(GF7(1), GF7(2)));
  EXPECT_EQ(test::ProjectivePoint(GF7(1), GF7(2), GF7(3)).ToAffine(),
            test::AffinePoint(GF7(5), GF7(3)));
}

TEST_F(ProjectivePointTest, ToJacobian) {
  EXPECT_EQ(test::ProjectivePoint(GF7(1), GF7(2), GF7(0)).ToJacobian(),
            test::JacobianPoint::Zero());
  EXPECT_EQ(test::ProjectivePoint(GF7(1), GF7(2), GF7(1)).ToJacobian(),
            test::JacobianPoint(GF7(1), GF7(2), GF7(1)));
  EXPECT_EQ(test::ProjectivePoint(GF7(1), GF7(2), GF7(3)).ToJacobian(),
            test::JacobianPoint(GF7(3), GF7(4), GF7(3)));
}

TEST_F(ProjectivePointTest, ToXYZZ) {
  EXPECT_EQ(test::ProjectivePoint(GF7(1), GF7(2), GF7(0)).ToXYZZ(),
            test::PointXYZZ::Zero());
  EXPECT_EQ(test::ProjectivePoint(GF7(1), GF7(2), GF7(3)).ToXYZZ(),
            test::PointXYZZ(GF7(3), GF7(4), GF7(2), GF7(6)));
}

TEST_F(ProjectivePointTest, BatchNormalize) {
  std::vector<test::ProjectivePoint> projective_points = {
      test::ProjectivePoint(GF7(1), GF7(2), GF7(0)),
      test::ProjectivePoint(GF7(1), GF7(2), GF7(1)),
      test::ProjectivePoint(GF7(1), GF7(2), GF7(3))};

  std::vector<test::AffinePoint> affine_points;
  affine_points.resize(2);
  ASSERT_FALSE(
      test::ProjectivePoint::BatchNormalize(projective_points, &affine_points));

  affine_points.resize(3);
  ASSERT_TRUE(
      test::ProjectivePoint::BatchNormalize(projective_points, &affine_points));

  std::vector<test::AffinePoint> expected_affine_points = {
      test::AffinePoint::Zero(), test::AffinePoint(GF7(1), GF7(2)),
      test::AffinePoint(GF7(5), GF7(3))};
  EXPECT_EQ(affine_points, expected_affine_points);
}

TEST_F(ProjectivePointTest, IsOnCurve) {
  test::ProjectivePoint invalid_point(GF7(1), GF7(2), GF7(1));
  EXPECT_FALSE(invalid_point.IsOnCurve());
  test::ProjectivePoint valid_point(GF7(3), GF7(2), GF7(1));
  EXPECT_TRUE(valid_point.IsOnCurve());
  valid_point = test::ProjectivePoint(GF7(3), GF7(5), GF7(1));
  EXPECT_TRUE(valid_point.IsOnCurve());
}

TEST_F(ProjectivePointTest, CreateFromX) {
  {
    std::optional<test::ProjectivePoint> p =
        test::ProjectivePoint::CreateFromX(GF7(3), /*pick_odd=*/true);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->y(), GF7(5));
  }
  {
    std::optional<test::ProjectivePoint> p =
        test::ProjectivePoint::CreateFromX(GF7(3), /*pick_odd=*/false);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->y(), GF7(2));
  }
  {
    std::optional<test::ProjectivePoint> p =
        test::ProjectivePoint::CreateFromX(GF7(1), /*pick_odd=*/false);
    ASSERT_FALSE(p.has_value());
  }
}

}  // namespace tachyon::math
