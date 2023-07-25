#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/curve_config.h"

namespace tachyon {
namespace math {

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

  EXPECT_TRUE(AffinePointTy::Zero().infinity());
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

TYPED_TEST(AffinePointTest, EqualityOperator) {
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
}

TYPED_TEST(AffinePointTest, ToJacobian) {
  using AffinePointTy = TypeParam;
  using JacobianPointTy = typename AffinePointTy::JacobianPointTy;
  using BaseField = typename AffinePointTy::BaseField;

  EXPECT_EQ(AffinePointTy::Identity().ToJacobian(), JacobianPointTy::Zero());
  AffinePointTy p(BaseField(3), BaseField(2));
  EXPECT_EQ(p.ToJacobian(),
            JacobianPointTy(BaseField(3), BaseField(2), BaseField(1)));
}

TYPED_TEST(AffinePointTest, IsOnCurve) {
  using AffinePointTy = TypeParam;
  using BaseField = typename AffinePointTy::BaseField;

  AffinePointTy invalid_point(BaseField(1), BaseField(2));
  EXPECT_FALSE(AffinePointTy::IsOnCurve(invalid_point));
  AffinePointTy valid_point(BaseField(3), BaseField(2));
  EXPECT_TRUE(AffinePointTy::IsOnCurve(valid_point));
  valid_point = AffinePointTy(BaseField(3), BaseField(5));
  EXPECT_TRUE(AffinePointTy::IsOnCurve(valid_point));
}

TYPED_TEST(AffinePointTest, MSM) {
  using AffinePointTy = TypeParam;
  using JacobianPointTy = typename AffinePointTy::JacobianPointTy;
  using BaseField = typename AffinePointTy::BaseField;
  using ScalarField = typename AffinePointTy::ScalarField;

  std::vector<AffinePointTy> bases = {
      {BaseField(5), BaseField(5)},
      {BaseField(3), BaseField(2)},
  };
  std::vector<ScalarField> scalars = {
      ScalarField(2),
      ScalarField(3),
  };
  JacobianPointTy expected = JacobianPointTy::Zero();
  for (size_t i = 0; i < bases.size(); ++i) {
    expected += bases[i].ToJacobian().ScalarMul(scalars[i].ToBigInt());
  }
  EXPECT_EQ(AffinePointTy::MSM(bases, scalars), expected);
}

}  // namespace math
}  // namespace tachyon
