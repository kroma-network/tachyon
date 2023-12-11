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
  using Poly = math::UnivariateDensePolynomial<Field, kMaxDegree>;
  using Evals = math::UnivariateEvaluations<Field, kMaxDegree>;
  using Domain = math::UnivariateEvaluationDomain<Field, kMaxDegree>;

  size_t D() const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->N() - 1;
  }

  // Commit to |poly| and populates |result| with the commitment.
  // Return false if the degree of |poly| exceeds |kMaxDegree|.
  [[nodiscard]] bool Commit(const Poly& poly, Commitment* result) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommit(poly, result);
  }

  // Commit to |poly| and populates |result| with the commitment.
  // Return false if the degree of |poly| exceeds |kMaxDegree|.
  [[nodiscard]] bool CommitLagrange(const Evals& evals,
                                    Commitment* result) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommitLagrange(evals, result);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_H_
