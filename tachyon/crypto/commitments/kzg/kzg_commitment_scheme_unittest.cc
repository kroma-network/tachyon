#include "tachyon/crypto/commitments/kzg/kzg_commitment_scheme.h"

#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::crypto {

namespace {

class KZGCommitmentSchemeTest : public testing::Test {
 public:
  using PCS = KZGCommitmentScheme<math::bn254::G1AffinePoint,
                                  math::bn254::G2AffinePoint,
                                  math::bn254::G1AffinePoint>;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

}  // namespace

TEST_F(KZGCommitmentSchemeTest, UnsafeSetup) {
  PCS kzg;
  ASSERT_TRUE(kzg.UnsafeSetup(32));

  EXPECT_EQ(kzg.K(), size_t{5});
  EXPECT_EQ(kzg.N(), size_t{32});
  EXPECT_EQ(kzg.g1_powers_of_tau().size(), size_t{32});
  EXPECT_EQ(kzg.g1_powers_of_tau_lagrange().size(), size_t{32});
}

TEST_F(KZGCommitmentSchemeTest, CommitLagrange) {
  using Field = math::bn254::G1AffinePoint::ScalarField;
  using DomainTy = math::UnivariateEvaluationDomain<Field, PCS::kMaxDegree>;
  using DensePoly = DomainTy::DensePoly;
  using Evals = DomainTy::Evals;

  PCS kzg;
  ASSERT_TRUE(kzg.UnsafeSetup(32));

  DensePoly poly = DensePoly::Random(31);

  math::bn254::G1AffinePoint commit;
  ASSERT_TRUE(kzg.Commit(poly, &commit));

  std::unique_ptr<DomainTy> domain =
      math::UnivariateEvaluationDomainFactory<Field, PCS::kMaxDegree>::Create(
          32);
  Evals poly_evals = domain->FFT(poly);

  math::bn254::G1AffinePoint commit_lagrange;
  ASSERT_TRUE(kzg.CommitLagrange(poly_evals, &commit_lagrange));

  EXPECT_EQ(commit, commit_lagrange);
}

TEST_F(KZGCommitmentSchemeTest, Downsize) {
  PCS kzg;
  ASSERT_TRUE(kzg.UnsafeSetup(64));
  ASSERT_FALSE(kzg.Downsize(kzg.N()));
  ASSERT_TRUE(kzg.Downsize(32));
  EXPECT_EQ(kzg.N(), size_t{32});
}

TEST_F(KZGCommitmentSchemeTest, Copyable) {
  PCS expected;
  ASSERT_TRUE(expected.UnsafeSetup(32));

  base::VectorBuffer write_buf;
  EXPECT_TRUE(write_buf.Write(expected));

  write_buf.set_buffer_offset(0);

  PCS value;
  EXPECT_TRUE(write_buf.Read(&value));

  EXPECT_EQ(expected.g1_powers_of_tau(), value.g1_powers_of_tau());
  EXPECT_EQ(expected.g1_powers_of_tau_lagrange(),
            value.g1_powers_of_tau_lagrange());
  EXPECT_EQ(expected.tau_g2(), value.tau_g2());
}

}  // namespace tachyon::crypto
