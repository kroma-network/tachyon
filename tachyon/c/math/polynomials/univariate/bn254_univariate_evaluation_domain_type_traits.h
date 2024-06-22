#ifndef TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATION_DOMAIN_TYPE_TRAITS_H_
#define TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATION_DOMAIN_TYPE_TRAITS_H_

#include "tachyon/c/base/type_traits_forward.h"
#include "tachyon/c/math/polynomials/constants.h"
#include "tachyon/c/math/polynomials/univariate/bn254_univariate_evaluation_domain.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/polynomials/univariate/univariate_evaluation_domain.h"

namespace tachyon::c::base {

template <>
struct TypeTraits<tachyon::math::UnivariateEvaluationDomain<
    tachyon::math::bn254::Fr, math::kMaxDegree>> {
  using CType = tachyon_bn254_univariate_evaluation_domain;
};

template <>
struct TypeTraits<tachyon_bn254_univariate_evaluation_domain> {
  using NativeType =
      tachyon::math::UnivariateEvaluationDomain<tachyon::math::bn254::Fr,
                                                math::kMaxDegree>;
};

}  // namespace tachyon::c::base

#endif  // TACHYON_C_MATH_POLYNOMIALS_UNIVARIATE_BN254_UNIVARIATE_EVALUATION_DOMAIN_TYPE_TRAITS_H_
