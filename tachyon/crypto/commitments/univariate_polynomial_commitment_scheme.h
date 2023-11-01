#ifndef TACHYON_CRYPTO_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_H_
#define TACHYON_CRYPTO_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_H_

#include <stddef.h>

#include "tachyon/crypto/commitments/vector_commitment_scheme.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::crypto {

template <typename Derived>
class UnivariatePolynomialCommitmentScheme
    : public VectorCommitmentScheme<Derived> {
 public:
  constexpr static size_t kMaxDegree =
      VectorCommitmentScheme<Derived>::kMaxSize - 1;

  using Field = typename VectorCommitmentScheme<Derived>::Field;
  using Commitment = typename VectorCommitmentScheme<Derived>::Commitment;

  // Commit to |poly| and populates |result| with the commitment.
  // Return false if the degree of |poly| exceeds |kMaxDegree|.
  [[nodiscard]] bool Commit(
      const math::UnivariateDensePolynomial<Field, kMaxDegree>& poly,
      Commitment* result) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommit(poly.coefficients().coefficients(), result);
  }

  // Commit to |poly| and populates |result| with the commitment.
  // Return false if the degree of |poly| exceeds |kMaxDegree|.
  [[nodiscard]] bool CommitLagrange(
      const math::UnivariateEvaluations<Field, kMaxDegree>& evals,
      Commitment* result) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommitLagrange(evals.evaluations(), result);
  }
};

}  // namespace tachyon::crypto
#endif  // TACHYON_CRYPTO_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_H_
