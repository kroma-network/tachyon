// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATION_DOMAIN_FACTORY_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATION_DOMAIN_FACTORY_H_

#include <memory>

#include "tachyon/math/polynomials/univariate/mixed_radix_evaluation_domain.h"
#include "tachyon/math/polynomials/univariate/radix2_evaluation_domain.h"

namespace tachyon::math {

template <typename F>
constexpr size_t MaxSizeForEvaluationDomainFactory() {
  size_t i = 1;
  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    for (size_t i = 0; i <= F::Config::kSmallSubgroupAdicity; ++i) {
      i *= F::Config::kSmallSubgroupBase;
    }
  }
  return i * (size_t{1} << F::Config::kTwoAdicity);
}

template <typename F, size_t N = MaxSizeForEvaluationDomainFactory<F>()>
class UnivariateEvaluationDomainFactory {
 public:
  // Construct a domain that is large enough for evaluations of a polynomial
  // having |num_coeffs| coefficients.
  static std::unique_ptr<UnivariateEvaluationDomain<F, N>> Create(
      size_t num_coeffs) {
    if (Radix2EvaluationDomain<F, N>::IsValidNumCoeffs(num_coeffs)) {
      return Radix2EvaluationDomain<F, N>::Create(num_coeffs);
    } else if (MixedRadixEvaluationDomain<F, N>::IsValidNumCoeffs(num_coeffs)) {
      return MixedRadixEvaluationDomain<F, N>::Create(num_coeffs);
    }
    NOTREACHED();
    return nullptr;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATION_DOMAIN_FACTORY_H_
