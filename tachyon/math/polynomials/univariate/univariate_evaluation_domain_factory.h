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
constexpr size_t MaxDegreeForEvaluationDomainFactory() {
  size_t i = 1;
  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    for (size_t i = 0; i <= F::Config::kSmallSubgroupAdicity; ++i) {
      i *= F::Config::kSmallSubgroupBase;
    }
  }
  return i * (size_t{1} << F::Config::kTwoAdicity) - 1;
}

template <typename F,
          size_t MaxDegree = MaxDegreeForEvaluationDomainFactory<F>()>
class UnivariateEvaluationDomainFactory {
 public:
  // Construct a domain that is large enough for evaluations of a polynomial
  // having |num_coeffs| coefficients.
  constexpr static std::unique_ptr<UnivariateEvaluationDomain<F, MaxDegree>>
  Create(size_t num_coeffs) {
    if (Radix2EvaluationDomain<F, MaxDegree>::IsValidNumCoeffs(num_coeffs)) {
      return Radix2EvaluationDomain<F, MaxDegree>::Create(num_coeffs);
    } else if (MixedRadixEvaluationDomain<F, MaxDegree>::IsValidNumCoeffs(
                   num_coeffs)) {
      return MixedRadixEvaluationDomain<F, MaxDegree>::Create(num_coeffs);
    }
    NOTREACHED();
    return nullptr;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_UNIVARIATE_EVALUATION_DOMAIN_FACTORY_H_
