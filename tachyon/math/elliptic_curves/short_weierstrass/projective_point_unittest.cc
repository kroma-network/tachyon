#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/curve_config.h"

namespace tachyon::math {

namespace {

template <typename ProjectivePointType>
class ProjectivePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { ProjectivePointType::Curve::Init(); }
};

}  // namespace

#if defined(TACHYON_GMP_BACKEND)
using ProjectivePointTypes =
    testing::Types<test::ProjectivePoint, test::ProjectivePointGmp>;
#else
using ProjectivePointTypes = testing::Types<test::ProjectivePoint>;
#endif
TYPED_TEST_SUITE(ProjectivePointTest, ProjectivePointTypes);

TYPED_TEST(ProjectivePointTest, IsZero) {
  using ProjectivePointTy = TypeParam;
  using BaseField = typename ProjectivePointTy::BaseField;

  EXPECT_TRUE(ProjectivePointTy::Zero().IsZero());
  EXPECT_FALSE(
      ProjectivePointTy(BaseField(1), BaseField(2), BaseField(1)).IsZero());
}

TYPED_TEST(ProjectivePointTest, Generator) {
  using ProjectivePointTy = TypeParam;
  using BaseField = typename ProjectivePointTy::BaseField;

  EXPECT_EQ(ProjectivePointTy::Generator(),
            ProjectivePointTy(ProjectivePointTy::Curve::Config::kGenerator.x,
                              ProjectivePointTy::Curve::Config::kGenerator.y,
                              BaseField::One()));
}

TYPED_TEST(ProjectivePointTest, Montgomery) {
  using ProjectivePointTy = TypeParam;

  ProjectivePointTy r = ProjectivePointTy::Random();
  EXPECT_EQ(r, ProjectivePointTy::FromMontgomery(r.ToMontgomery()));
}

TYPED_TEST(ProjectivePointTest, Random) {
  using ProjectivePointTy = TypeParam;

  bool success = false;
  ProjectivePointTy r = ProjectivePointTy::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != ProjectivePointTy::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TYPED_TEST(ProjectivePointTest, EqualityOperators) {
  using ProjectivePointTy = TypeParam;
  using BaseField = typename ProjectivePointTy::BaseField;

  {
    SCOPED_TRACE("p.IsZero() && p2.IsZero()");
    ProjectivePointTy p(BaseField(1), BaseField(2), BaseField(0));
    ProjectivePointTy p2(BaseField(3), BaseField(4), BaseField(0));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }

  {
    SCOPED_TRACE("!p.IsZero() && p2.IsZero()");
    ProjectivePointTy p(BaseField(1), BaseField(2), BaseField(1));
    ProjectivePointTy p2(BaseField(3), BaseField(4), BaseField(0));
    EXPECT_TRUE(p != p2);
    EXPECT_TRUE(p2 != p);
  }

  {
    SCOPED_TRACE("other");
    ProjectivePointTy p(BaseField(1), BaseField(2), BaseField(3));
    ProjectivePointTy p2(BaseField(1), BaseField(2), BaseField(3));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }
}

TYPED_TEST(ProjectivePointTest, AdditiveGroupOperators) {
  using ProjectivePointTy = TypeParam;
  using AffinePointTy = typename ProjectivePointTy::AffinePointTy;
  using BaseField = typename ProjectivePointTy::BaseField;
  using ScalarField = typename AffinePointTy::ScalarField;

  ProjectivePointTy pp = ProjectivePointTy::CreateChecked(
      BaseField(5), BaseField(5), BaseField(1));
  ProjectivePointTy pp2 = ProjectivePointTy::CreateChecked(
      BaseField(3), BaseField(2), BaseField(1));
  ProjectivePointTy pp3 = ProjectivePointTy::CreateChecked(
      BaseField(3), BaseField(5), BaseField(1));
  ProjectivePointTy pp4 = ProjectivePointTy::CreateChecked(
      BaseField(6), BaseField(5), BaseField(1));
  AffinePointTy ap = pp.ToAffine();
  AffinePointTy ap2 = pp2.ToAffine();

  EXPECT_EQ(pp + pp2, pp3);
  EXPECT_EQ(pp - pp3, -pp2);
  EXPECT_EQ(pp + pp, pp4);
  EXPECT_EQ(pp - pp4, -pp);

  {
    ProjectivePointTy pp_tmp = pp;
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
    ProjectivePointTy pp_tmp = pp;
    pp_tmp.DoubleInPlace();
    EXPECT_EQ(pp_tmp, pp4);
  }

  EXPECT_EQ(pp.Negative(),
            ProjectivePointTy(BaseField(5), BaseField(2), BaseField(1)));
  {
    ProjectivePointTy pp_tmp = pp;
    pp_tmp.NegInPlace();
    EXPECT_EQ(pp_tmp,
              ProjectivePointTy(BaseField(5), BaseField(2), BaseField(1)));
  }

  EXPECT_EQ(pp * ScalarField(2), pp4);
  EXPECT_EQ(ScalarField(2) * pp, pp4);
  EXPECT_EQ(pp *= ScalarField(2), pp4);
}

TYPED_TEST(ProjectivePointTest, ScalarMulOperator) {
  using ProjectivePointTy = TypeParam;
  using AffinePointTy = typename ProjectivePointTy::AffinePointTy;
  using BaseField = typename ProjectivePointTy::BaseField;
  using ScalarField = typename ProjectivePointTy::ScalarField;

  std::vector<AffinePointTy> points;
  for (size_t i = 0; i < 7; ++i) {
    points.push_back(
        (ScalarField(i) * ProjectivePointTy::Generator()).ToAffine());
  }

  EXPECT_THAT(points,
              testing::UnorderedElementsAreArray(std::vector<AffinePointTy>{
                  AffinePointTy(BaseField(0), BaseField(0), true),
                  AffinePointTy(BaseField(3), BaseField(2)),
                  AffinePointTy(BaseField(5), BaseField(2)),
                  AffinePointTy(BaseField(6), BaseField(2)),
                  AffinePointTy(BaseField(3), BaseField(5)),
                  AffinePointTy(BaseField(5), BaseField(5)),
                  AffinePointTy(BaseField(6), BaseField(5))}));
}

TYPED_TEST(ProjectivePointTest, ToAffine) {
  using ProjectivePointTy = TypeParam;
  using AffinePointTy = typename ProjectivePointTy::AffinePointTy;
  using BaseField = typename ProjectivePointTy::BaseField;

  EXPECT_EQ(
      ProjectivePointTy(BaseField(1), BaseField(2), BaseField(0)).ToAffine(),
      AffinePointTy::Zero());
  EXPECT_EQ(
      ProjectivePointTy(BaseField(1), BaseField(2), BaseField(1)).ToAffine(),
      AffinePointTy(BaseField(1), BaseField(2)));
  EXPECT_EQ(
      ProjectivePointTy(BaseField(1), BaseField(2), BaseField(3)).ToAffine(),
      AffinePointTy(BaseField(5), BaseField(3)));
}

TYPED_TEST(ProjectivePointTest, ToJacobian) {
  using ProjectivePointTy = TypeParam;
  using JacobianPointTy = typename ProjectivePointTy::JacobianPointTy;
  using BaseField = typename ProjectivePointTy::BaseField;

  EXPECT_EQ(
      ProjectivePointTy(BaseField(1), BaseField(2), BaseField(0)).ToJacobian(),
      JacobianPointTy::Zero());
  EXPECT_EQ(
      ProjectivePointTy(BaseField(1), BaseField(2), BaseField(1)).ToJacobian(),
      JacobianPointTy(BaseField(1), BaseField(2), BaseField(1)));
  EXPECT_EQ(
      ProjectivePointTy(BaseField(1), BaseField(2), BaseField(3)).ToJacobian(),
      JacobianPointTy(BaseField(3), BaseField(4), BaseField(3)));
}

TYPED_TEST(ProjectivePointTest, ToXYZZ) {
  using ProjectivePointTy = TypeParam;
  using PointXYZZTy = typename ProjectivePointTy::PointXYZZTy;
  using BaseField = typename ProjectivePointTy::BaseField;

  EXPECT_EQ(
      ProjectivePointTy(BaseField(1), BaseField(2), BaseField(0)).ToXYZZ(),
      PointXYZZTy::Zero());
  EXPECT_EQ(
      ProjectivePointTy(BaseField(1), BaseField(2), BaseField(3)).ToXYZZ(),
      PointXYZZTy(BaseField(3), BaseField(4), BaseField(2), BaseField(6)));
}

TYPED_TEST(ProjectivePointTest, IsOnCurve) {
  using ProjectivePointTy = TypeParam;
  using BaseField = typename ProjectivePointTy::BaseField;

  ProjectivePointTy invalid_point(BaseField(1), BaseField(2), BaseField(1));
  EXPECT_FALSE(invalid_point.IsOnCurve());
  ProjectivePointTy valid_point(BaseField(3), BaseField(2), BaseField(1));
  EXPECT_TRUE(valid_point.IsOnCurve());
  valid_point = ProjectivePointTy(BaseField(3), BaseField(5), BaseField(1));
  EXPECT_TRUE(valid_point.IsOnCurve());
}

}  // namespace tachyon::math
