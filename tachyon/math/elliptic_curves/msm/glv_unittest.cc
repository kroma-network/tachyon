#include "tachyon/math/elliptic_curves/msm/glv.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls/bls12_381/g1.h"

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

using PointTypes =
    testing::Types<bls12_381::G1AffinePoint, bls12_381::G1ProjectivePoint,
                   bls12_381::G1JacobianPoint, bls12_381::G1PointXYZZ>;
TYPED_TEST_SUITE(GLVTest, PointTypes);

TYPED_TEST(GLVTest, Endomorphism) {
  using PointTy = TypeParam;

  EXPECT_TRUE(
      GLV<PointTy>::EndomorphismCoefficient().Pow(BigInt<1>(3)).IsOne());
  PointTy base = PointTy::Random();

  PointTy expected;
  auto lambda = GLV<PointTy>::Lambda().ToBigInt();
  if constexpr (std::is_same_v<PointTy, bls12_381::G1AffinePoint>) {
    expected = base.ToJacobian().ScalarMul(lambda).ToAffine();
  } else {
    expected = base.ScalarMul(lambda);
  }
  EXPECT_EQ(expected, PointTy::Endomorphism(base));
}

TYPED_TEST(GLVTest, Decompose) {
  using PointTy = TypeParam;

  bls12_381::Fr scalar = bls12_381::Fr::Random();
  auto result = GLV<PointTy>::Decompose(scalar);
  bls12_381::Fr k1 = bls12_381::Fr::FromMpzClass(result.k1.abs_value);
  bls12_381::Fr k2 = bls12_381::Fr::FromMpzClass(result.k2.abs_value);
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
  using ReturnTy = typename PointTraits<PointTy>::AdditionResultTy;

  PointTy base = PointTy::Random();
  bls12_381::Fr scalar = bls12_381::Fr::Random();
  ReturnTy expected;
  if constexpr (std::is_same_v<PointTy, bls12_381::G1AffinePoint>) {
    expected = base.ToJacobian().ScalarMul(scalar.ToBigInt());
  } else {
    expected = base.ScalarMul(scalar.ToBigInt());
  }
  EXPECT_EQ(GLV<PointTy>::Mul(base, scalar), expected);
}

}  // namespace tachyon::math
