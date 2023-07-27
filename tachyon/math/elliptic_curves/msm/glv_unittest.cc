#include "tachyon/math/elliptic_curves/msm/glv.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls/bls12_381/g1.h"

namespace tachyon {
namespace math {

namespace {

using Config = bls12_381::G1CurveConfig<bls12_381::FqGmp, bls12_381::FrGmp>;
using Curve = SWCurve<Config>;

template <typename PointTy>
class GLVTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    Curve::Init();
    GLV<Curve>::Init();
  }
};

}  // namespace

using PointTypes =
    testing::Types<bls12_381::G1AffinePointGmp, bls12_381::G1ProjectivePointGmp,
                   bls12_381::G1JacobianPointGmp, bls12_381::G1PointXYZZGmp>;
TYPED_TEST_SUITE(GLVTest, PointTypes);

TYPED_TEST(GLVTest, Endomorphism) {
  using PointTy = TypeParam;

  EXPECT_TRUE(GLV<Curve>::EndomorphismCoefficient().Pow(BigInt<1>(3)).IsOne());
  PointTy base = PointTy::Random();

  PointTy expected;
  auto lambda = GLV<Curve>::Lambda().ToBigInt();
  if constexpr (std::is_same_v<PointTy, bls12_381::G1AffinePointGmp>) {
    expected = base.ToJacobian().ScalarMul(lambda).ToAffine();
  } else {
    expected = base.ScalarMul(lambda);
  }
  EXPECT_EQ(expected, PointTy::Endomorphism(base));
}

TEST(GLVTestDecompose, Decompose) {
  Curve::Init();
  GLV<Curve>::Init();

  bls12_381::FrGmp scalar = bls12_381::FrGmp::Random();
  auto result = GLV<Curve>::Decompose(scalar);
  bls12_381::FrGmp k1(result.k1.abs_value);
  bls12_381::FrGmp k2(result.k2.abs_value);
  if (result.k1.sign == Sign::kNegative) {
    k1.NegInPlace();
  }
  if (result.k2.sign == Sign::kNegative) {
    k2.NegInPlace();
  }
  EXPECT_EQ(scalar, k1 + GLV<Curve>::Lambda() * k2);
}

TYPED_TEST(GLVTest, Mul) {
  using PointTy = TypeParam;
  using PointRetTy = typename GLVTraits<PointTy>::ReturnType;

  PointTy base = PointTy::Random();
  bls12_381::FrGmp scalar = bls12_381::FrGmp::Random();
  PointRetTy expected;
  if constexpr (std::is_same_v<PointTy, bls12_381::G1AffinePointGmp>) {
    expected = base.ToJacobian().ScalarMul(scalar.ToBigInt());
  } else {
    expected = base.ScalarMul(scalar.ToBigInt());
  }
  EXPECT_EQ(GLV<Curve>::Mul(base, scalar), expected);
}

}  // namespace math
}  // namespace tachyon
