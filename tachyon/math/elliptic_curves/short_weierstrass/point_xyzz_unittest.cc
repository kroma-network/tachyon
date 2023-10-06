#include "tachyon/math/elliptic_curves/short_weierstrass/point_xyzz.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/projective_point.h"
#include "tachyon/math/elliptic_curves/short_weierstrass/test/curve_config.h"

namespace tachyon::math {

namespace {

template <typename PointXYZZType>
class PointXYZZTest : public testing::Test {
 public:
  static void SetUpTestSuite() { PointXYZZType::Curve::Init(); }
};

}  // namespace

#if defined(TACHYON_GMP_BACKEND)
using PointXYZZTypes = testing::Types<test::PointXYZZ, test::PointXYZZGmp>;
#else
using PointXYZZTypes = testing::Types<test::PointXYZZ>;
#endif
TYPED_TEST_SUITE(PointXYZZTest, PointXYZZTypes);

TYPED_TEST(PointXYZZTest, IsZero) {
  using PointXYZZTy = TypeParam;
  using BaseField = typename PointXYZZTy::BaseField;

  EXPECT_TRUE(
      PointXYZZTy(BaseField(1), BaseField(2), BaseField(0), BaseField(0))
          .IsZero());
  EXPECT_FALSE(
      PointXYZZTy(BaseField(1), BaseField(2), BaseField(1), BaseField(0))
          .IsZero());
  EXPECT_TRUE(
      PointXYZZTy(BaseField(1), BaseField(2), BaseField(0), BaseField(1))
          .IsZero());
}

TYPED_TEST(PointXYZZTest, Generator) {
  using PointXYZZTy = TypeParam;
  using BaseField = typename PointXYZZTy::BaseField;

  EXPECT_EQ(PointXYZZTy::Generator(),
            PointXYZZTy(PointXYZZTy::Curve::Config::kGenerator.x,
                        PointXYZZTy::Curve::Config::kGenerator.y,
                        BaseField::One(), BaseField::One()));
}

TYPED_TEST(PointXYZZTest, Montgomery) {
  using PointXYZZTy = TypeParam;

  PointXYZZTy r = PointXYZZTy::Random();
  EXPECT_EQ(r, PointXYZZTy::FromMontgomery(r.ToMontgomery()));
}

TYPED_TEST(PointXYZZTest, Random) {
  using PointXYZZTy = TypeParam;

  bool success = false;
  PointXYZZTy r = PointXYZZTy::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != PointXYZZTy::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TYPED_TEST(PointXYZZTest, EqualityOperators) {
  using PointXYZZTy = TypeParam;
  using BaseField = typename PointXYZZTy::BaseField;

  {
    SCOPED_TRACE("p.IsZero() && p2.IsZero()");
    PointXYZZTy p(BaseField(1), BaseField(2), BaseField(0), BaseField(0));
    PointXYZZTy p2(BaseField(3), BaseField(4), BaseField(0), BaseField(0));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }

  {
    SCOPED_TRACE("!p.IsZero() && p2.IsZero()");
    PointXYZZTy p(BaseField(1), BaseField(2), BaseField(1), BaseField(0));
    PointXYZZTy p2(BaseField(3), BaseField(4), BaseField(0), BaseField(0));
    EXPECT_TRUE(p != p2);
    EXPECT_TRUE(p2 != p);
  }

  {
    SCOPED_TRACE("other");
    PointXYZZTy p(BaseField(1), BaseField(2), BaseField(2), BaseField(6));
    PointXYZZTy p2(BaseField(1), BaseField(2), BaseField(2), BaseField(6));
    EXPECT_TRUE(p == p2);
    EXPECT_TRUE(p2 == p);
  }
}

TYPED_TEST(PointXYZZTest, AdditiveGroupOperators) {
  using PointXYZZTy = TypeParam;
  using AffinePointTy = typename PointXYZZTy::AffinePointTy;
  using BaseField = typename PointXYZZTy::BaseField;
  using ScalarField = typename AffinePointTy::ScalarField;

  PointXYZZTy p = PointXYZZTy::CreateChecked(BaseField(5), BaseField(5),
                                             BaseField(1), BaseField(1));
  PointXYZZTy p2 = PointXYZZTy::CreateChecked(BaseField(3), BaseField(2),
                                              BaseField(1), BaseField(1));
  PointXYZZTy p3 = PointXYZZTy::CreateChecked(BaseField(3), BaseField(5),
                                              BaseField(1), BaseField(1));
  PointXYZZTy p4 = PointXYZZTy::CreateChecked(BaseField(6), BaseField(5),
                                              BaseField(1), BaseField(1));
  AffinePointTy ap = p.ToAffine();
  AffinePointTy ap2 = p2.ToAffine();

  EXPECT_EQ(p + p2, p3);
  EXPECT_EQ(p - p3, -p2);
  EXPECT_EQ(p + p, p4);
  EXPECT_EQ(p - p4, -p);

  {
    PointXYZZTy p_tmp = p;
    p_tmp += p2;
    EXPECT_EQ(p_tmp, p3);
    p_tmp -= p2;
    EXPECT_EQ(p_tmp, p);
  }

  EXPECT_EQ(p + ap2, p3);
  EXPECT_EQ(p + ap, p4);
  EXPECT_EQ(p - p3, -p2);
  EXPECT_EQ(p - p4, -p);

  EXPECT_EQ(p.Double(), p4);
  {
    PointXYZZTy p_tmp = p;
    p_tmp.DoubleInPlace();
    EXPECT_EQ(p_tmp, p4);
  }

  EXPECT_EQ(p.Negative(), PointXYZZTy(BaseField(5), BaseField(2), BaseField(1),
                                      BaseField(1)));
  {
    PointXYZZTy p_tmp = p;
    p_tmp.NegInPlace();
    EXPECT_EQ(p_tmp, PointXYZZTy(BaseField(5), BaseField(2), BaseField(1),
                                 BaseField(1)));
  }

  EXPECT_EQ(p * ScalarField(2), p4);
  EXPECT_EQ(ScalarField(2) * p, p4);
  EXPECT_EQ(p *= ScalarField(2), p4);
}

TYPED_TEST(PointXYZZTest, ScalarMulOperator) {
  using PointXYZZTy = TypeParam;
  using AffinePointTy = typename PointXYZZTy::AffinePointTy;
  using BaseField = typename PointXYZZTy::BaseField;
  using ScalarField = typename PointXYZZTy::ScalarField;

  std::vector<AffinePointTy> points;
  for (size_t i = 0; i < 7; ++i) {
    points.push_back((ScalarField(i) * PointXYZZTy::Generator()).ToAffine());
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

TYPED_TEST(PointXYZZTest, ToAffine) {
  using PointXYZZTy = TypeParam;
  using AffinePointTy = typename PointXYZZTy::AffinePointTy;
  using BaseField = typename PointXYZZTy::BaseField;

  EXPECT_EQ(PointXYZZTy(BaseField(1), BaseField(2), BaseField(0), BaseField(0))
                .ToAffine(),
            AffinePointTy::Zero());
  EXPECT_EQ(PointXYZZTy(BaseField(1), BaseField(2), BaseField(1), BaseField(1))
                .ToAffine(),
            AffinePointTy(BaseField(1), BaseField(2)));
  EXPECT_EQ(PointXYZZTy(BaseField(1), BaseField(2), BaseField(2), BaseField(6))
                .ToAffine(),
            AffinePointTy(BaseField(4), BaseField(5)));
}

TYPED_TEST(PointXYZZTest, ToProjective) {
  using PointXYZZTy = TypeParam;
  using ProjectivePointTy = typename PointXYZZTy::ProjectivePointTy;
  using BaseField = typename PointXYZZTy::BaseField;

  EXPECT_EQ(PointXYZZTy(BaseField(1), BaseField(2), BaseField(0), BaseField(0))
                .ToProjective(),
            ProjectivePointTy::Zero());
  EXPECT_EQ(PointXYZZTy(BaseField(1), BaseField(2), BaseField(1), BaseField(1))
                .ToProjective(),
            ProjectivePointTy(BaseField(1), BaseField(2), BaseField(1)));
  EXPECT_EQ(PointXYZZTy(BaseField(1), BaseField(2), BaseField(2), BaseField(6))
                .ToProjective(),
            ProjectivePointTy(BaseField(6), BaseField(4), BaseField(5)));
}

TYPED_TEST(PointXYZZTest, ToJacobian) {
  using PointXYZZTy = TypeParam;
  using JacobianPointTy = typename PointXYZZTy::JacobianPointTy;
  using BaseField = typename PointXYZZTy::BaseField;

  EXPECT_EQ(PointXYZZTy(BaseField(1), BaseField(2), BaseField(0), BaseField(0))
                .ToJacobian(),
            JacobianPointTy::Zero());
  EXPECT_EQ(PointXYZZTy(BaseField(1), BaseField(2), BaseField(1), BaseField(1))
                .ToJacobian(),
            JacobianPointTy(BaseField(1), BaseField(2), BaseField(1)));
  EXPECT_EQ(PointXYZZTy(BaseField(1), BaseField(2), BaseField(2), BaseField(6))
                .ToJacobian(),
            JacobianPointTy(BaseField(2), BaseField(2), BaseField(5)));
}

}  // namespace tachyon::math
