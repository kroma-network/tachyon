#include "tachyon/crypto/commitments/kzg/kzg.h"

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g2.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::crypto {

namespace {

class KZGTest : public testing::Test {
 public:
  using Field = math::bn254::G1AffinePoint::ScalarField;
  using KZGParamsTy =
      KZGParams<math::bn254::G1AffinePoint, math::bn254::G2AffinePoint>;

  static constexpr size_t MaxDegree = size_t{1} << Field::Config::kTwoAdicity;

  static void SetUpTestSuite() { math::bn254::G1Curve::Init(); }
};

}  // namespace

TEST_F(KZGTest, UnsafeSetup) {
  KZGParamsTy kzg;
  ASSERT_TRUE(kzg.UnsafeSetup(5));

  EXPECT_EQ(kzg.k(), size_t{5});
  EXPECT_EQ(kzg.n(), size_t{32});
  EXPECT_EQ(kzg.g1_powers_of_tau().size(), size_t{32});
  EXPECT_EQ(kzg.g1_powers_of_tau_lagrange().size(), size_t{32});
}

TEST_F(KZGTest, CommitLagrange) {
  KZGParamsTy kzg;
  ASSERT_TRUE(kzg.UnsafeSetup(5));

  math::UnivariateEvaluationDomain<Field, MaxDegree>::DensePoly poly =
      math::UnivariateEvaluationDomain<Field, MaxDegree>::DensePoly::Random(31);

  math::bn254::G1AffinePoint commit;
  ASSERT_TRUE(kzg.Commit(poly, &commit));

  std::unique_ptr<math::UnivariateEvaluationDomain<Field, MaxDegree>> domain =
      math::UnivariateEvaluationDomainFactory<Field, MaxDegree>::Create(32);
  math::UnivariateEvaluationDomain<Field, MaxDegree>::Evals poly_evals =
      domain->FFT(poly);

  math::bn254::G1AffinePoint commit_lagrange;
  ASSERT_TRUE(kzg.CommitLagrange(poly_evals, &commit_lagrange));

  EXPECT_EQ(commit, commit_lagrange);
}

TEST_F(KZGTest, Downsize) {
  KZGParamsTy kzg;
  ASSERT_TRUE(kzg.UnsafeSetup(8));

  kzg.Downsize(5);
  EXPECT_EQ(kzg.n(), size_t{32});
}

}  // namespace tachyon::crypto
