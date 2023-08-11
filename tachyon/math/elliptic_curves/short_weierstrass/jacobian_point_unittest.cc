#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/curve_config.h"

namespace tachyon::math {

namespace {

template <typename JacobianPointType>
class JacobianPointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { JacobianPointType::Curve::Init(); }
};

}  // namespace

#if defined(TACHYON_GMP_BACKEND)
using JacobianPointTypes =
    testing::Types<test::JacobianPoint, test::JacobianPointGmp>;
#else
using JacobianPointTypes = testing::Types<test::JacobianPoint>;
#endif
TYPED_TEST_SUITE(JacobianPointTest, JacobianPointTypes);

TYPED_TEST(JacobianPointTest, IsZero) {
  using JacobianPointTy = TypeParam;
  using BaseField = typename JacobianPointTy::BaseField;

  EXPECT_TRUE(JacobianPointTy::Zero().IsZero());
  EXPECT_FALSE(
      JacobianPointTy(BaseField(1), BaseField(2), BaseField(1)).IsZero());
}

TYPED_TEST(JacobianPointTest, Montgomery) {
  using JacobianPointTy = TypeParam;

  JacobianPointTy r = JacobianPointTy::Random();
  EXPECT_EQ(r, JacobianPointTy::FromMontgomery(r.ToMontgomery()));
}

TYPED_TEST(JacobianPointTest, Random) {
  using JacobianPointTy = TypeParam;

  bool success = false;
  JacobianPointTy r = JacobianPointTy::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != JacobianPointTy::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TYPED_TEST(JacobianPointTest, EqualityOperators) {
  using JacobianPointTy = TypeParam;
  using BaseField = typename JacobianPointTy::BaseField;

  {
    SCOPED_TRACE("p.IsZero() && p2.IsZero()");
    JacobianPointTy p(BaseField(1), BaseField(2), BaseField(0));
    JacobianPointTy p2(BaseField(3), BaseField(4), BaseField(0));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }

  {
    SCOPED_TRACE("!p.IsZero() && p2.IsZero()");
    JacobianPointTy p(BaseField(1), BaseField(2), BaseField(1));
    JacobianPointTy p2(BaseField(3), BaseField(4), BaseField(0));
    EXPECT_TRUE(p != p2);
    EXPECT_TRUE(p2 != p);
  }

  {
    SCOPED_TRACE("other");
    JacobianPointTy p(BaseField(1), BaseField(2), BaseField(3));
    JacobianPointTy p2(BaseField(1), BaseField(2), BaseField(3));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }
}

TYPED_TEST(JacobianPointTest, AdditiveGroupOperators) {
  using JacobianPointTy = TypeParam;
  using AffinePointTy = typename JacobianPointTy::AffinePointTy;
  using BaseField = typename JacobianPointTy::BaseField;

  JacobianPointTy jp =
      JacobianPointTy::CreateChecked(BaseField(5), BaseField(5), BaseField(1));
  JacobianPointTy jp2 =
      JacobianPointTy::CreateChecked(BaseField(3), BaseField(2), BaseField(1));
  JacobianPointTy jp3 =
      JacobianPointTy::CreateChecked(BaseField(3), BaseField(5), BaseField(1));
  JacobianPointTy jp4 =
      JacobianPointTy::CreateChecked(BaseField(6), BaseField(5), BaseField(1));
  AffinePointTy ap = jp.ToAffine();
  AffinePointTy ap2 = jp2.ToAffine();

  EXPECT_EQ(jp + jp2, jp3);
  EXPECT_EQ(jp - jp3, -jp2);
  EXPECT_EQ(jp + jp, jp4);
  EXPECT_EQ(jp - jp4, -jp);

  {
    JacobianPointTy jp_tmp = jp;
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
    JacobianPointTy jp_tmp = jp;
    jp_tmp.DoubleInPlace();
    EXPECT_EQ(jp_tmp, jp4);
  }

  EXPECT_EQ(jp.Negative(),
            JacobianPointTy(BaseField(5), BaseField(2), BaseField(1)));
  {
    JacobianPointTy jp_tmp = jp;
    jp_tmp.NegInPlace();
    EXPECT_EQ(jp_tmp,
              JacobianPointTy(BaseField(5), BaseField(2), BaseField(1)));
  }
}

TYPED_TEST(JacobianPointTest, ScalarMulOperator) {
  using JacobianPointTy = TypeParam;
  using AffinePointTy = typename JacobianPointTy::AffinePointTy;
  using BaseField = typename JacobianPointTy::BaseField;
  using ScalarField = typename JacobianPointTy::ScalarField;

  std::vector<AffinePointTy> points;
  for (size_t i = 0; i < 7; ++i) {
    points.push_back(
        (ScalarField(i) * JacobianPointTy::Curve::Generator()).ToAffine());
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

TYPED_TEST(JacobianPointTest, ToAffine) {
  using JacobianPointTy = TypeParam;
  using AffinePointTy = typename JacobianPointTy::AffinePointTy;
  using BaseField = typename JacobianPointTy::BaseField;

  EXPECT_EQ(
      JacobianPointTy(BaseField(1), BaseField(2), BaseField(0)).ToAffine(),
      AffinePointTy::Zero());
  EXPECT_EQ(
      JacobianPointTy(BaseField(1), BaseField(2), BaseField(1)).ToAffine(),
      AffinePointTy(BaseField(1), BaseField(2)));
  EXPECT_EQ(
      JacobianPointTy(BaseField(1), BaseField(2), BaseField(3)).ToAffine(),
      AffinePointTy(BaseField(4), BaseField(5)));
}

TYPED_TEST(JacobianPointTest, ToProjective) {
  using JacobianPointTy = TypeParam;
  using ProjectivePointTy = typename JacobianPointTy::ProjectivePointTy;
  using BaseField = typename JacobianPointTy::BaseField;

  EXPECT_EQ(
      JacobianPointTy(BaseField(1), BaseField(2), BaseField(0)).ToProjective(),
      ProjectivePointTy::Zero());
  EXPECT_EQ(
      JacobianPointTy(BaseField(1), BaseField(2), BaseField(3)).ToProjective(),
      ProjectivePointTy(BaseField(3), BaseField(2), BaseField(6)));
}

TYPED_TEST(JacobianPointTest, ToXYZZ) {
  using JacobianPointTy = TypeParam;
  using PointXYZZTy = typename JacobianPointTy::PointXYZZTy;
  using BaseField = typename JacobianPointTy::BaseField;

  EXPECT_EQ(JacobianPointTy(BaseField(1), BaseField(2), BaseField(0)).ToXYZZ(),
            PointXYZZTy::Zero());
  EXPECT_EQ(
      JacobianPointTy(BaseField(1), BaseField(2), BaseField(3)).ToXYZZ(),
      PointXYZZTy(BaseField(1), BaseField(2), BaseField(2), BaseField(6)));
}

TYPED_TEST(JacobianPointTest, MSM) {
  using JacobianPointTy = TypeParam;
  using BaseField = typename JacobianPointTy::BaseField;
  using ScalarField = typename JacobianPointTy::ScalarField;

  std::vector<JacobianPointTy> bases = {
      {BaseField(5), BaseField(5), BaseField(1)},
      {BaseField(3), BaseField(2), BaseField(1)},
  };
  std::vector<ScalarField> scalars = {
      ScalarField(2),
      ScalarField(3),
  };
  JacobianPointTy expected = JacobianPointTy::Zero();
  for (size_t i = 0; i < bases.size(); ++i) {
    expected += bases[i].ScalarMul(scalars[i].ToBigInt());
  }
  EXPECT_EQ(JacobianPointTy::MSM(bases, scalars), expected);
}

}  // namespace tachyon::math
