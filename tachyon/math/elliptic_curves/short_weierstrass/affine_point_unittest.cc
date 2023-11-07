#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/sw_curve_config.h"

namespace tachyon::math {

namespace {

template <typename AffinePointType>
class AffinePointTest : public testing::Test {
 public:
  static void SetUpTestSuite() { AffinePointType::Curve::Init(); }
};

}  // namespace

#if defined(TACHYON_GMP_BACKEND)
using AffinePointTypes =
    testing::Types<test::AffinePoint, test::AffinePointGmp>;
#else
using AffinePointTypes = testing::Types<test::AffinePoint>;
#endif
TYPED_TEST_SUITE(AffinePointTest, AffinePointTypes);

TYPED_TEST(AffinePointTest, Zero) {
  using AffinePointTy = TypeParam;
  using BaseField = typename AffinePointTy::BaseField;

  EXPECT_TRUE(AffinePointTy::Zero().infinity());
  EXPECT_TRUE(AffinePointTy::Zero().IsZero());
  EXPECT_FALSE(AffinePointTy(BaseField(1), BaseField(2)).IsZero());
}

TYPED_TEST(AffinePointTest, Generator) {
  using AffinePointTy = TypeParam;

  EXPECT_EQ(AffinePointTy::Generator(),
            AffinePointTy(AffinePointTy::Curve::Config::kGenerator.x,
                          AffinePointTy::Curve::Config::kGenerator.y));
}

TYPED_TEST(AffinePointTest, Montgomery) {
  using AffinePointTy = TypeParam;

  AffinePointTy r = AffinePointTy::Random();
  while (r.infinity()) {
    r = AffinePointTy::Random();
  }
  EXPECT_EQ(r, AffinePointTy::FromMontgomery(r.ToMontgomery()));
}

TYPED_TEST(AffinePointTest, Random) {
  using AffinePointTy = TypeParam;

  bool success = false;
  AffinePointTy r = AffinePointTy::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != AffinePointTy::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TYPED_TEST(AffinePointTest, EqualityOperators) {
  using AffinePointTy = TypeParam;
  using BaseField = typename AffinePointTy::BaseField;

  AffinePointTy p(BaseField(1), BaseField(2));
  AffinePointTy p2(BaseField(3), BaseField(4));
  EXPECT_TRUE(p == p);
  EXPECT_TRUE(p != p2);
}

TYPED_TEST(AffinePointTest, AdditiveGroupOperators) {
  using AffinePointTy = TypeParam;
  using JacobianPointTy = typename AffinePointTy::JacobianPointTy;
  using BaseField = typename AffinePointTy::BaseField;
  using ScalarField = typename AffinePointTy::ScalarField;

  AffinePointTy ap = AffinePointTy::CreateChecked(BaseField(5), BaseField(5));
  AffinePointTy ap2 = AffinePointTy::CreateChecked(BaseField(3), BaseField(2));
  AffinePointTy ap3 = AffinePointTy::CreateChecked(BaseField(3), BaseField(5));
  AffinePointTy ap4 = AffinePointTy::CreateChecked(BaseField(6), BaseField(5));
  JacobianPointTy jp = ap.ToJacobian();
  JacobianPointTy jp2 = ap2.ToJacobian();
  JacobianPointTy jp3 = ap3.ToJacobian();
  JacobianPointTy jp4 = ap4.ToJacobian();

  EXPECT_EQ(ap + ap2, jp3);
  EXPECT_EQ(ap + ap, jp4);
  EXPECT_EQ(ap3 - ap2, jp);
  EXPECT_EQ(ap4 - ap, jp);

  EXPECT_EQ(ap + jp2, jp3);
  EXPECT_EQ(ap + jp, jp4);
  EXPECT_EQ(ap - jp3, -jp2);
  EXPECT_EQ(ap - jp4, -jp);

  EXPECT_EQ(ap.Double(), jp4);
  EXPECT_EQ(ap.DoubleProjective(), ap4.ToProjective());
  EXPECT_EQ(ap.DoubleXYZZ(), ap4.ToXYZZ());

  EXPECT_EQ(-ap, AffinePointTy(BaseField(5), BaseField(2)));
  {
    AffinePointTy ap_tmp = ap;
    ap_tmp.NegInPlace();
    EXPECT_EQ(ap_tmp, AffinePointTy(BaseField(5), BaseField(2)));
  }

  EXPECT_EQ(ap * ScalarField(2), jp4);
  EXPECT_EQ(ScalarField(2) * ap, jp4);
}

TYPED_TEST(AffinePointTest, ToProjective) {
  using AffinePointTy = TypeParam;
  using ProjectivePointTy = typename AffinePointTy::ProjectivePointTy;
  using BaseField = typename AffinePointTy::BaseField;

  EXPECT_EQ(AffinePointTy::Zero().ToProjective(), ProjectivePointTy::Zero());
  AffinePointTy p(BaseField(3), BaseField(2));
  EXPECT_EQ(p.ToProjective(),
            ProjectivePointTy(BaseField(3), BaseField(2), BaseField(1)));
}

TYPED_TEST(AffinePointTest, ToJacobian) {
  using AffinePointTy = TypeParam;
  using JacobianPointTy = typename AffinePointTy::JacobianPointTy;
  using BaseField = typename AffinePointTy::BaseField;

  EXPECT_EQ(AffinePointTy::Zero().ToJacobian(), JacobianPointTy::Zero());
  AffinePointTy p(BaseField(3), BaseField(2));
  EXPECT_EQ(p.ToJacobian(),
            JacobianPointTy(BaseField(3), BaseField(2), BaseField(1)));
}

TYPED_TEST(AffinePointTest, ToPointXYZZ) {
  using AffinePointTy = TypeParam;
  using PointXYZZTy = typename AffinePointTy::PointXYZZTy;
  using BaseField = typename AffinePointTy::BaseField;

  EXPECT_EQ(AffinePointTy::Zero().ToXYZZ(), PointXYZZTy::Zero());
  AffinePointTy p(BaseField(3), BaseField(2));
  EXPECT_EQ(p.ToXYZZ(), PointXYZZTy(BaseField(3), BaseField(2), BaseField(1),
                                    BaseField(1)));
}

TYPED_TEST(AffinePointTest, IsOnCurve) {
  using AffinePointTy = TypeParam;
  using BaseField = typename AffinePointTy::BaseField;

  AffinePointTy invalid_point(BaseField(1), BaseField(2));
  EXPECT_FALSE(invalid_point.IsOnCurve());
  AffinePointTy valid_point(BaseField(3), BaseField(2));
  EXPECT_TRUE(valid_point.IsOnCurve());
  valid_point = AffinePointTy(BaseField(3), BaseField(5));
  EXPECT_TRUE(valid_point.IsOnCurve());
}

TYPED_TEST(AffinePointTest, CreateFromX) {
  using AffinePointTy = TypeParam;
  using BaseField = typename AffinePointTy::BaseField;

  {
    std::optional<AffinePointTy> p =
        AffinePointTy::CreateFromX(BaseField(3), /*pick_odd=*/true);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->y(), BaseField(5));
  }
  {
    std::optional<AffinePointTy> p =
        AffinePointTy::CreateFromX(BaseField(3), /*pick_odd=*/false);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->y(), BaseField(2));
  }
  {
    std::optional<AffinePointTy> p =
        AffinePointTy::CreateFromX(BaseField(1), /*pick_odd=*/false);
    ASSERT_FALSE(p.has_value());
  }
}

}  // namespace tachyon::math
