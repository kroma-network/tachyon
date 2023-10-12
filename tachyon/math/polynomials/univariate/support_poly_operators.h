// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_SUPPORT_POLY_OPERATORS_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_SUPPORT_POLY_OPERATORS_H_

#include "tachyon/math/polynomials/support_poly_operators.h"

#define SUPPORTS_POLY_OPERATOR(Name)                                     \
  template <typename Coefficients, typename L, typename R>               \
  struct SupportsPoly##Name<                                             \
      Coefficients, L, R,                                                \
      decltype(void(UnivariatePolynomialOp<Coefficients>::Name(          \
          std::declval<const L&>(), std::declval<const R&>())))>         \
      : std::true_type {};                                               \
                                                                         \
  template <typename Coefficients, typename L, typename R>               \
  struct SupportsPoly##Name##InPlace<                                    \
      Coefficients, L, R,                                                \
      decltype(void(UnivariatePolynomialOp<Coefficients>::Name##InPlace( \
          std::declval<L&>(), std::declval<const R&>())))> : std::true_type {}

namespace tachyon::math::internal {

template <typename Coefficients>
class UnivariatePolynomialOp;

SUPPORTS_POLY_OPERATOR(Add);
SUPPORTS_POLY_OPERATOR(Sub);
SUPPORTS_POLY_OPERATOR(Mul);
SUPPORTS_POLY_OPERATOR(Div);
SUPPORTS_POLY_OPERATOR(Mod);

}  // namespace tachyon::math::internal

#undef SUPPORTS_POLY_OPERATOR

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_SUPPORT_POLY_OPERATORS_H_
