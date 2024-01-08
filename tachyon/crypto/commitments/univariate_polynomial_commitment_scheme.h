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
  using TranscriptReader =
      typename VectorCommitmentSchemeTraits<Derived>::TranscriptReader;
  using TranscriptWriter =
      typename VectorCommitmentSchemeTraits<Derived>::TranscriptWriter;
  using Poly = math::UnivariateDensePolynomial<Field, kMaxDegree>;
  using Evals = math::UnivariateEvaluations<Field, kMaxDegree>;
  using Domain = math::UnivariateEvaluationDomain<Field, kMaxDegree>;

  size_t D() const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->N() - 1;
  }

  // Commit to |poly| and populates |commitment|.
  // Return false if the degree of |poly| exceeds |kMaxDegree|.
  template <typename T = Derived,
            std::enable_if_t<!VectorCommitmentSchemeTraits<
                T>::kIsCommitInteractive>* = nullptr>
  [[nodiscard]] bool Commit(const Poly& poly, Commitment* commitment) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommit(poly, commitment);
  }

  // Commit to |poly| with a |transcript_writer| and populates |commitment|.
  // Return false if the degree of |poly| exceeds |kMaxDegree|.
  template <typename T = Derived, std::enable_if_t<VectorCommitmentSchemeTraits<
                                      T>::IsCommitInteractive>* = nullptr>
  [[nodiscard]] bool Commit(const Poly& poly,
                            TranscriptWriter* transcript_writer,
                            Commitment* commitment) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommit(poly, transcript_writer, commitment);
  }

  // Commit to |evals| and populates |commitment|.
  // Return false if the degree of |evals| exceeds |kMaxDegree|.
  template <typename T = Derived,
            std::enable_if_t<!VectorCommitmentSchemeTraits<
                T>::kIsCommitInteractive>* = nullptr>
  [[nodiscard]] bool CommitLagrange(const Evals& evals,
                                    Commitment* commitment) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommitLagrange(evals, commitment);
  }

  // Commit to |evals| with a |transcript_writer| and populates |commitment|.
  // Return false if the degree of |evals| exceeds |kMaxDegree|.
  template <typename T = Derived, std::enable_if_t<VectorCommitmentSchemeTraits<
                                      T>::kIsCommitInteractive>* = nullptr>
  [[nodiscard]] bool CommitLagrange(const Evals& evals,
                                    TranscriptWriter* transcript_writer,
                                    Commitment* commitment) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoCommitLagrange(evals, transcript_writer, commitment);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_H_
