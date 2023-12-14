#include "tachyon/math/elliptic_curves/pairing/pairing.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls12/bls12_381/bls12_381.h"
#include "tachyon/math/elliptic_curves/bn/bn254/bn254.h"

namespace tachyon::math {

template <typename CurveType>
class PairingTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    using G1Curve = typename CurveType::G1Curve;
    using G2Curve = typename CurveType::G2Curve;

    G1Curve::Init();
    G2Curve::Init();
    CurveType::Init();
  }
};

using CurveTypes = testing::Types<bn254::BN254Curve, bls12_381::BLS12_381Curve>;
TYPED_TEST_SUITE(PairingTest, CurveTypes);

TYPED_TEST(PairingTest, Bilinearity) {
  using Curve = TypeParam;
  using G1Curve = typename Curve::G1Curve;
  using G1AffinePointTy = typename G1Curve::AffinePointTy;
  using G2Curve = typename Curve::G2Curve;
  using G2AffinePointTy = typename G2Curve::AffinePointTy;
  using G2Prepared = typename Curve::G2Prepared;
  using ScalarField = typename G1Curve::ScalarField;
  using Fp12Ty = typename Curve::Fp12Ty;

  G1AffinePointTy g1 = G1AffinePointTy::Random();
  G2AffinePointTy g2 = G2AffinePointTy::Random();
  ScalarField a = ScalarField::Random();
  ScalarField b = ScalarField::Random();

  Fp12Ty result;
  {
    std::vector<G1AffinePointTy> g1s = {(a * b * g1).ToAffine()};
    std::vector<G2Prepared> g2s = {G2Prepared::From(g2)};
    result = Pairing<Curve>(g1s, g2s);
  }

  Fp12Ty result2;
  {
    std::vector<G1AffinePointTy> g1s = {(a * g1).ToAffine()};
    std::vector<G2Prepared> g2s = {G2Prepared::From((b * g2).ToAffine())};
    result2 = Pairing<Curve>(g1s, g2s);
  }

  EXPECT_EQ(result, result2);

  Fp12Ty result3;
  {
    std::vector<G1AffinePointTy> g1s = {(b * g1).ToAffine()};
    std::vector<G2Prepared> g2s = {G2Prepared::From((a * g2).ToAffine())};
    result3 = Pairing<Curve>(g1s, g2s);
  }

  EXPECT_EQ(result, result3);

  Fp12Ty result4;
  {
    std::vector<G1AffinePointTy> g1s = {g1};
    std::vector<G2Prepared> g2s = {G2Prepared::From((a * b * g2).ToAffine())};
    result4 = Pairing<Curve>(g1s, g2s);
  }

  EXPECT_EQ(result, result4);
}

}  // namespace tachyon::math
