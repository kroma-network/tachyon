#ifndef TACHYON_CRYPTO_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_H_
#define TACHYON_CRYPTO_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_H_

#include <stddef.h>

#include "tachyon/crypto/commitments/vector_commitment_scheme.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluations.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::crypto {

template <typename C>
class UnivariatePolynomialCommitmentScheme : public VectorCommitmentScheme<C> {
 public:
  constexpr static size_t kMaxDegree = VectorCommitmentScheme<C>::kMaxSize - 1;

  using Field = typename VectorCommitmentScheme<C>::Field;
  using Commitment = typename VectorCommitmentScheme<C>::Commitment;

  // Commit to |poly| and populates |result| with the commitment.
  // Return false if the degree of |poly| exceeds |kMaxDegree|.
  [[nodiscard]] bool Commit(
      const math::UnivariateDensePolynomial<Field, kMaxDegree>& poly,
      Commitment* result) const {
    const C* c = static_cast<const C*>(this);
    return c->DoCommit(poly.coefficients().coefficients(), result);
  }

  // Commit to |poly| and populates |result| with the commitment.
  // Return false if the degree of |poly| exceeds |kMaxDegree|.
  [[nodiscard]] bool CommitLagrange(
      const math::UnivariateEvaluations<Field, kMaxDegree>& evals,
      Commitment* result) const {
    const C* c = static_cast<const C*>(this);
    return c->DoCommitLagrange(evals.evaluations(), result);
  }
};

}  // namespace tachyon::crypto
#endif  // TACHYON_CRYPTO_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_H_
