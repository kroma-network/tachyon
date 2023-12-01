#ifndef TACHYON_ZK_BASE_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_EXTENSION_H_
#define TACHYON_ZK_BASE_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_EXTENSION_H_

#include <stddef.h>

#include "tachyon/crypto/commitments/univariate_polynomial_commitment_scheme.h"
#include "tachyon/math/base/rational_field.h"
#include "tachyon/zk/base/commitments/univariate_polynomial_commitment_scheme_extension_traits.h"

namespace tachyon::zk {

template <typename Derived>
class UnivariatePolynomialCommitmentSchemeExtension
    : public crypto::UnivariatePolynomialCommitmentScheme<Derived> {
 public:
  constexpr static size_t kMaxDegree =
      crypto::UnivariatePolynomialCommitmentScheme<Derived>::kMaxDegree;
  constexpr static size_t kMaxExtendedDegree =
      UnivariatePolynomialCommitmentSchemeExtensionTraits<
          Derived>::kMaxExtendedDegree;

  using Field =
      typename crypto::UnivariatePolynomialCommitmentScheme<Derived>::Field;
  using RationalField = math::RationalField<Field>;
  using RationalPoly =
      math::UnivariateDensePolynomial<RationalField, kMaxDegree>;
  using RationalEvals = math::UnivariateEvaluations<RationalField, kMaxDegree>;

  using ExtendedDomain =
      math::UnivariateEvaluationDomain<Field, kMaxExtendedDegree>;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_COMMITMENTS_UNIVARIATE_POLYNOMIAL_COMMITMENT_SCHEME_EXTENSION_H_
