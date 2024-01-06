#include "tachyon/math/elliptic_curves/pairing/pairing.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls12/bls12_381/bls12_381.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"

namespace tachyon::math {

template <typename Curve>
class PairingTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    using G1Curve = typename Curve::G1Curve;
    using G2Curve = typename Curve::G2Curve;

    G1Curve::Init();
    G2Curve::Init();
    Curve::Init();
  }
};

using CurveTypes = testing::Types<bn254::BN254Curve, bls12_381::BLS12_381Curve>;
TYPED_TEST_SUITE(PairingTest, CurveTypes);

TYPED_TEST(PairingTest, Bilinearity) {
  using Curve = TypeParam;
  using G1Curve = typename Curve::G1Curve;
  using G1AffinePoint = typename G1Curve::AffinePoint;
  using G2Curve = typename Curve::G2Curve;
  using G2AffinePoint = typename G2Curve::AffinePoint;
  using G2Prepared = typename Curve::G2Prepared;
  using ScalarField = typename G1Curve::ScalarField;
  using Fp12 = typename Curve::Fp12;

  G1AffinePoint g1 = G1AffinePoint::Random();
  G2AffinePoint g2 = G2AffinePoint::Random();
  ScalarField a = ScalarField::Random();
  ScalarField b = ScalarField::Random();

  Fp12 result;
  {
    G1AffinePoint g1s[] = {(a * b * g1).ToAffine()};
    G2Prepared g2s[] = {G2Prepared::From(g2)};
    result = Pairing<Curve>(g1s, g2s);
  }

  Fp12 result2;
  {
    G1AffinePoint g1s[] = {(a * g1).ToAffine()};
    G2Prepared g2s[] = {G2Prepared::From((b * g2).ToAffine())};
    result2 = Pairing<Curve>(g1s, g2s);
  }

  EXPECT_EQ(result, result2);

  Fp12 result3;
  {
    G1AffinePoint g1s[] = {(b * g1).ToAffine()};
    G2Prepared g2s[] = {G2Prepared::From((a * g2).ToAffine())};
    result3 = Pairing<Curve>(g1s, g2s);
  }

  EXPECT_EQ(result, result3);

  Fp12 result4;
  {
    G1AffinePoint g1s[] = {g1};
    G2Prepared g2s[] = {G2Prepared::From((a * b * g2).ToAffine())};
    result4 = Pairing<Curve>(g1s, g2s);
  }

  EXPECT_EQ(result, result4);
}

}  // namespace tachyon::math
