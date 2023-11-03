#include "tachyon/zk/base/blinded_polynomial_commitment.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/crypto/commitments/kzg/kzg_commitment_scheme.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/g1.h"
#include "tachyon/math/elliptic_curves/bls/bls12_381/g2.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::zk {

namespace {

class BlindedPolynomialCommitmentTest : public testing::Test {
 public:
  constexpr static size_t kMaxDegree = (size_t{1} << 5) - 1;
  constexpr static size_t kNumCoeffs = kMaxDegree + 1;

  using PCS =
      crypto::KZGCommitmentScheme<math::bls12_381::G1AffinePoint,
                                  math::bls12_381::G2AffinePoint, kMaxDegree>;
  using Field = PCS::Field;
  using Evals = PCS::Evals;
  using Commitment = PCS::Commitment;
  using Domain = PCS::Domain;

  static void SetUpTestSuite() { math::bls12_381::G1Curve::Init(); }
};

}  // namespace

TEST_F(BlindedPolynomialCommitmentTest, CommitEvalsWithBlind) {
  // setting domain
  std::unique_ptr<Domain> domain =
      math::UnivariateEvaluationDomainFactory<Field, kMaxDegree>::Create(
          kNumCoeffs);

  // setting random polynomial
  Evals evals = Evals::Random();

  // setting commitment scheme
  PCS pcs;
  ASSERT_TRUE(pcs.UnsafeSetup(kNumCoeffs));

  // setting struct to get output
  BlindedPolynomialCommitment<PCS> out;
  ASSERT_TRUE(CommitEvalsWithBlind(domain.get(), evals, pcs, &out));

  EXPECT_EQ(out.poly(), domain->IFFT(evals));
  Commitment expected_commit;
  ASSERT_TRUE(pcs.CommitLagrange(evals, &expected_commit));
  EXPECT_EQ(out.commitment(), expected_commit);
}

}  // namespace tachyon::zk
