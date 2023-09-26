#include "tachyon/math/elliptic_curves/msm/glv.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls/bls12_381/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/point_conversions.h"

namespace tachyon::math {

namespace {

template <typename PointTy>
class GLVTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    PointTy::Curve::Init();
    GLV<PointTy>::Init();
  }
};

}  // namespace

// This iterates all the points in bls12_381 to test whether GLV works on
// various points.
using PointTypes =
    testing::Types<bls12_381::G1AffinePoint, bls12_381::G1ProjectivePoint,
                   bls12_381::G1JacobianPoint, bls12_381::G1PointXYZZ,
                   bn254::G1JacobianPoint>;
TYPED_TEST_SUITE(GLVTest, PointTypes);

TYPED_TEST(GLVTest, Endomorphism) {
  using PointTy = TypeParam;
  using ReturnTy =
      typename internal::AdditiveSemigroupTraits<PointTy>::ReturnTy;

  EXPECT_TRUE(
      GLV<PointTy>::EndomorphismCoefficient().Pow(BigInt<1>(3)).IsOne());
  PointTy base = PointTy::Random();
  EXPECT_EQ(base * GLV<PointTy>::Lambda(),
            ConvertPoint<ReturnTy>(PointTy::Endomorphism(base)));
}

TYPED_TEST(GLVTest, Decompose) {
  using PointTy = TypeParam;
  using ScalarField = typename PointTy::ScalarField;

  ScalarField scalar = ScalarField::Random();
  auto result = GLV<PointTy>::Decompose(scalar);
  ScalarField k1 = ScalarField::FromMpzClass(result.k1.abs_value);
  ScalarField k2 = ScalarField::FromMpzClass(result.k2.abs_value);
  if (result.k1.sign == Sign::kNegative) {
    k1.NegInPlace();
  }
  if (result.k2.sign == Sign::kNegative) {
    k2.NegInPlace();
  }
  EXPECT_EQ(scalar, k1 + GLV<PointTy>::Lambda() * k2);
}

TYPED_TEST(GLVTest, Mul) {
  using PointTy = TypeParam;
  using ScalarField = typename PointTy::ScalarField;

  PointTy base = PointTy::Random();
  ScalarField scalar = ScalarField::Random();
  EXPECT_EQ(GLV<PointTy>::Mul(base, scalar), base * scalar);
}

}  // namespace tachyon::math
