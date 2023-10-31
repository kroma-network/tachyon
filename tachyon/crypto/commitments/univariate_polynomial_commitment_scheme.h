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
  constexpr static size_t kMaxSize = VectorCommitmentScheme<C>::kMaxSize;

  using Field = typename VectorCommitmentScheme<C>::Field;
  using ResultTy = typename VectorCommitmentScheme<C>::ResultTy;

  // Commit to |poly| and populates |result| with the commitment.
  // Return false if the size of |poly| exceeds |kMaxSize|.
  [[nodiscard]] bool Commit(
      const math::UnivariateDensePolynomial<Field, kMaxSize>& poly,
      ResultTy* result) const {
    const C* c = static_cast<const C*>(this);
    return c->DoCommit(poly.coefficients().coefficients(), result);
  }

  // Commit to |poly| and populates |result| with the commitment.
  // Return false if the size of |poly| exceeds |kMaxSize|.
  [[nodiscard]] bool CommitLagrange(
      const math::UnivariateEvaluations<Field, kMaxSize>& evals,
      ResultTy* result) const {
    const C* c = static_cast<const C*>(this);
    return c->DoCommitLagrange(evals.evaluations(), result);
  }
};

}  // namespace tachyon::crypto
#endif  // TACHYON_CRYPTO_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_H_
