#include "tachyon/math/elliptic_curves/msm/glv.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls12/bls12_381/g1.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_381/g2.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"

namespace tachyon::math {

namespace {

template <typename Point>
class GLVTest : public testing::Test {
 public:
  static void SetUpTestSuite() { Point::Curve::Init(); }
};

}  // namespace

// This iterates all the points in bls12_381 G1 to test whether GLV works on
// various points.
using PointTypes =
    testing::Types<bls12_381::G1AffinePoint, bls12_381::G1ProjectivePoint,
                   bls12_381::G1JacobianPoint, bls12_381::G1PointXYZZ,
                   bls12_381::G2JacobianPoint, bn254::G1JacobianPoint,
                   bn254::G2JacobianPoint>;
TYPED_TEST_SUITE(GLVTest, PointTypes);

TYPED_TEST(GLVTest, Endomorphism) {
  using Point = TypeParam;
  using RetPoint = typename internal::AdditiveSemigroupTraits<Point>::ReturnTy;

  EXPECT_TRUE(Point::Curve::Config::kEndomorphismCoefficient.Pow(3).IsOne());
  Point base = Point::Random();
  EXPECT_EQ(base * Point::Curve::Config::kLambda,
            ConvertPoint<RetPoint>(Point::Endomorphism(base)));
}

TYPED_TEST(GLVTest, Decompose) {
  using Point = TypeParam;
  using ScalarField = typename Point::ScalarField;

  ScalarField scalar = ScalarField::Random();
  auto result = GLV<Point>::Decompose(scalar);
  ScalarField k1 = ScalarField::FromMpzClass(result.k1.abs_value);
  ScalarField k2 = ScalarField::FromMpzClass(result.k2.abs_value);
  if (result.k1.sign == Sign::kNegative) {
    k1.NegInPlace();
  }
  if (result.k2.sign == Sign::kNegative) {
    k2.NegInPlace();
  }
  EXPECT_EQ(scalar, k1 + Point::Curve::Config::kLambda * k2);
}

TYPED_TEST(GLVTest, Mul) {
  using Point = TypeParam;
  using ScalarField = typename Point::ScalarField;

  Point base = Point::Random();
  ScalarField scalar = ScalarField::Random();
  EXPECT_EQ(GLV<Point>::Mul(base, scalar), base * scalar);
}

}  // namespace tachyon::math
