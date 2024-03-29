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

  // Commit to |poly| and stores the commitment in |batch_commitments_| at
  // |index| if |batch_mode| is true. Return false if the degree of |poly|
  // exceeds |kMaxDegree|. It terminates when |batch_mode| is false.
  template <typename T = Derived, std::enable_if_t<VectorCommitmentSchemeTraits<
                                      T>::kSupportsBatchMode>* = nullptr>
  [[nodiscard]] bool Commit(const Poly& poly, size_t index) {
    Derived* derived = static_cast<Derived*>(this);
    CHECK(derived->GetBatchMode());
    return derived->DoCommit(poly, derived->batch_commitment_state(), index);
  }

  // Commit to |evals| and populates |result| with the commitment.
  // Return false if the degree of |evals| exceeds |kMaxDegree|.
  [[nodiscard]] bool CommitLagrange(const Evals& evals,
                                    Commitment* result) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommitLagrange(evals, result);
  }

  // Commit to |evals| and stores the commitment in |batch_commitments_| at
  // |index| if |batch_mode| is true. Return false if the degree of |evals|
  // exceeds |kMaxDegree|. It terminates when |batch_mode| is false.
  template <typename T = Derived, std::enable_if_t<VectorCommitmentSchemeTraits<
                                      T>::kSupportsBatchMode>* = nullptr>
  [[nodiscard]] bool CommitLagrange(const Evals& evals, size_t index) {
    Derived* derived = static_cast<Derived*>(this);
    CHECK(derived->GetBatchMode());
    return derived->DoCommitLagrange(evals, derived->batch_commitment_state(),
                                     index);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_H_
