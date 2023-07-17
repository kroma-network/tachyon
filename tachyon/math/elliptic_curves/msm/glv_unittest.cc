#include "tachyon/math/elliptic_curves/msm/glv.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls/bls12_381/curve_config.h"

namespace tachyon {
namespace math {

namespace {

class GLVTest : public ::testing::Test {
 public:
  GLVTest() { bls12_381::CurveConfig::Init(); }
  GLVTest(const GLVTest&) = delete;
  GLVTest& operator=(const GLVTest&) = delete;
  ~GLVTest() override = default;
};

}  // namespace

TEST_F(GLVTest, Endomorphism) {
  EXPECT_TRUE(bls12_381::CurveConfig::EndomorphismCoefficient()
                  .Pow(BigInt<1>(3))
                  .IsOne());
  bls12_381::G1JacobianPoint base = bls12_381::G1JacobianPoint::Random();
  EXPECT_EQ(base.ScalarMul(bls12_381::CurveConfig::Lambda().ToBigInt()),
            bls12_381::CurveConfig::Endomorphism(base));
}

TEST_F(GLVTest, Decompose) {
  bls12_381::Fr scalar = bls12_381::Fr::Random();
  auto result = GLV<bls12_381::CurveConfig>::Decompose(scalar);
  bls12_381::Fr k1(result.k1.abs_value);
  bls12_381::Fr k2(result.k2.abs_value);
  if (result.k1.sign == Sign::kNegative) {
    k1.NegInPlace();
  }
  if (result.k2.sign == Sign::kNegative) {
    k2.NegInPlace();
  }
  EXPECT_EQ(scalar, k1 + bls12_381::CurveConfig::Lambda() * k2);
}

TEST_F(GLVTest, Mul) {
  bls12_381::G1JacobianPoint base = bls12_381::G1JacobianPoint::Random();
  bls12_381::Fr scalar = bls12_381::Fr::Random();
  EXPECT_EQ(GLV<bls12_381::CurveConfig>::Mul(base, scalar),
            base.ScalarMul(scalar.ToBigInt()));

  EXPECT_EQ(GLV<bls12_381::CurveConfig>::Mul(base.ToAffine(), scalar),
            base.ScalarMul(scalar.ToBigInt()));
}

}  // namespace math
}  // namespace tachyon
