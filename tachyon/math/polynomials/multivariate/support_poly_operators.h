#ifndef TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_SUPPORT_POLY_OPERATORS_H_
#define TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_SUPPORT_POLY_OPERATORS_H_

#include "tachyon/math/polynomials/support_poly_operators.h"

#define SUPPORTS_POLY_OPERATOR(Name)                                       \
  template <typename Coefficients, typename L, typename R>                 \
  struct SupportsPoly##Name<                                               \
      Coefficients, L, R,                                                  \
      decltype(void(MultivariatePolynomialOp<Coefficients>::Name(          \
          std::declval<const L&>(), std::declval<const R&>())))>           \
      : std::true_type {};                                                 \
                                                                           \
  template <typename Coefficients, typename L, typename R>                 \
  struct SupportsPoly##Name##InPlace<                                      \
      Coefficients, L, R,                                                  \
      decltype(void(MultivariatePolynomialOp<Coefficients>::Name##InPlace( \
          std::declval<L&>(), std::declval<const R&>())))> : std::true_type {}

namespace tachyon::math::internal {

template <typename Coefficients, typename SFINAE = void>
class MultivariatePolynomialOp;

SUPPORTS_POLY_OPERATOR(Add);
SUPPORTS_POLY_OPERATOR(Sub);

}  // namespace tachyon::math::internal

#undef SUPPORTS_POLY_OPERATOR

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_SUPPORT_POLY_OPERATORS_H_
