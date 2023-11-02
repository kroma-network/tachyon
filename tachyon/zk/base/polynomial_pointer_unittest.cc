#include "tachyon/zk/base/polynomial_pointer.h"

#include "gtest/gtest.h"

#include "tachyon/crypto/commitments/kzg/kzg_commitment_scheme.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"

namespace tachyon::zk {

class PolynomialPointerTest : public testing::Test {
 public:
  constexpr static size_t kMaxDegree = 7;

  using PCS =
      crypto::KZGCommitmentScheme<math::bn254::G1AffinePoint,
                                  math::bn254::G2AffinePoint, kMaxDegree,
                                  math::bn254::G1AffinePoint>;
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

TEST_F(PolynomialPointerTest, ProverQuery) {
  Poly poly = Poly::Random(kMaxDegree);
  F blind = F::Random();
  PolynomialPointer<PCS> pointer(poly, blind);
  EXPECT_EQ(&pointer.poly(), &poly);

  // Different instances with same reference.
  PolynomialPointer<PCS> pointer2(poly, blind);
  EXPECT_EQ(pointer, pointer2);

  // Different instances with different references. (but same value)
  F blind2 = blind;
  PolynomialPointer<PCS> pointer3(poly, blind2);
  EXPECT_NE(pointer, pointer3);
}

}  // namespace tachyon::zk
