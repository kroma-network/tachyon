#include "tachyon/math/elliptic_curves/msm/glv.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls/bls12_381/curve_config.h"

namespace tachyon {
namespace math {

namespace {

using CurveConfig = bls12_381::CurveConfig<bls12_381::FqGmp, bls12_381::FrGmp>;

class GLVTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() { CurveConfig::Init(); }
};

}  // namespace

TEST_F(GLVTest, Endomorphism) {
  EXPECT_TRUE(CurveConfig::EndomorphismCoefficient().Pow(BigInt<1>(3)).IsOne());
  bls12_381::G1JacobianPointGmp base = bls12_381::G1JacobianPointGmp::Random();
  EXPECT_EQ(base.ScalarMul(CurveConfig::Lambda().ToBigInt()),
            CurveConfig::Endomorphism(base));
}

TEST_F(GLVTest, Decompose) {
  bls12_381::FrGmp scalar = bls12_381::FrGmp::Random();
  auto result = GLV<CurveConfig>::Decompose(scalar);
  bls12_381::FrGmp k1(result.k1.abs_value);
  bls12_381::FrGmp k2(result.k2.abs_value);
  if (result.k1.sign == Sign::kNegative) {
    k1.NegInPlace();
  }
  if (result.k2.sign == Sign::kNegative) {
    k2.NegInPlace();
  }
  EXPECT_EQ(scalar, k1 + CurveConfig::Lambda() * k2);
}

TEST_F(GLVTest, Mul) {
  bls12_381::G1JacobianPointGmp base = bls12_381::G1JacobianPointGmp::Random();
  bls12_381::FrGmp scalar = bls12_381::FrGmp::Random();
  EXPECT_EQ(GLV<CurveConfig>::Mul(base, scalar),
            base.ScalarMul(scalar.ToBigInt()));

  EXPECT_EQ(GLV<CurveConfig>::Mul(base.ToAffine(), scalar),
            base.ScalarMul(scalar.ToBigInt()));
}

}  // namespace math
}  // namespace tachyon
