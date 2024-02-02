#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain.h"

#include "gtest/gtest.h"

#include "tachyon/math/base/rational_field.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain_factory.h"

namespace tachyon::math {

namespace {

constexpr size_t kDegree = 5;

class UnivariateEvaluationDomainTest : public testing::Test {
 public:
  constexpr static size_t kMaxDegree = SIZE_MAX;

  using Domain = UnivariateEvaluationDomain<bn254::Fr, kMaxDegree>;
  using RationalEvals =
      UnivariateEvaluations<RationalField<bn254::Fr>, kMaxDegree>;

  static void SetUpTestSuite() { bn254::Fr::Init(); }

  void SetUp() override {
    domain_ = tachyon_bn254_univariate_evaluation_domain_create(kDegree);
  }

  void TearDown() override {
    tachyon_bn254_univariate_evaluation_domain_destroy(domain_);
  }

 protected:
  tachyon_bn254_univariate_evaluation_domain* domain_;
};

}  // namespace

TEST_F(UnivariateEvaluationDomainTest, EmptyEvals) {
  Domain::Evals cpp_evals =
      reinterpret_cast<Domain*>(domain_)->Empty<Domain::Evals>();
  tachyon_bn254_univariate_evaluations* evals =
      tachyon_bn254_univariate_evaluation_domain_empty_evals(domain_);
  EXPECT_EQ(cpp_evals, reinterpret_cast<Domain::Evals&>(*evals));
  tachyon_bn254_univariate_evaluations_destroy(evals);
}

TEST_F(UnivariateEvaluationDomainTest, EmptyPoly) {
  Domain::DensePoly cpp_poly =
      reinterpret_cast<Domain*>(domain_)->Empty<Domain::DensePoly>();
  tachyon_bn254_univariate_dense_polynomial* poly =
      tachyon_bn254_univariate_evaluation_domain_empty_poly(domain_);
  EXPECT_EQ(cpp_poly, reinterpret_cast<Domain::DensePoly&>(*poly));
  tachyon_bn254_univariate_dense_polynomial_destroy(poly);
}

TEST_F(UnivariateEvaluationDomainTest, EmptyRationalEvals) {
  RationalEvals cpp_evals =
      reinterpret_cast<Domain*>(domain_)->Empty<RationalEvals>();
  tachyon_bn254_univariate_rational_evaluations* evals =
      tachyon_bn254_univariate_evaluation_domain_empty_rational_evals(domain_);
  EXPECT_EQ(cpp_evals, reinterpret_cast<RationalEvals&>(*evals));
  tachyon_bn254_univariate_rational_evaluations_destroy(evals);
}

TEST_F(UnivariateEvaluationDomainTest, FFT) {
  Domain::DensePoly poly = Domain::DensePoly::Random(kDegree);
  tachyon_bn254_univariate_evaluations* evals =
      tachyon_bn254_univariate_evaluation_domain_fft(
          domain_,
          reinterpret_cast<const tachyon_bn254_univariate_dense_polynomial*>(
              &poly));
  EXPECT_EQ(Domain::Create(kDegree)->FFT(poly),
            reinterpret_cast<Domain::Evals&>(*evals));
  tachyon_bn254_univariate_evaluations_destroy(evals);
}

TEST_F(UnivariateEvaluationDomainTest, IFFT) {
  Domain::Evals evals = Domain::Evals::Random(kDegree);
  tachyon_bn254_univariate_dense_polynomial* poly =
      tachyon_bn254_univariate_evaluation_domain_ifft(
          domain_,
          reinterpret_cast<const tachyon_bn254_univariate_evaluations*>(
              &evals));
  EXPECT_EQ(Domain::Create(kDegree)->IFFT(evals),
            reinterpret_cast<Domain::DensePoly&>(*poly));
  tachyon_bn254_univariate_dense_polynomial_destroy(poly);
}

}  // namespace tachyon::math
