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

#define SUPPORTS_MLE_OPERATOR(Name)                                     \
  template <typename Evaluations, typename L, typename R>               \
  struct SupportsPoly##Name<                                            \
      Evaluations, L, R,                                                \
      decltype(void(MultilinearExtensionOp<Evaluations>::Name(          \
          std::declval<const L&>(), std::declval<const R&>())))>        \
      : std::true_type {};                                              \
                                                                        \
  template <typename Evaluations, typename L, typename R>               \
  struct SupportsPoly##Name##InPlace<                                   \
      Evaluations, L, R,                                                \
      decltype(void(MultilinearExtensionOp<Evaluations>::Name##InPlace( \
          std::declval<L&>(), std::declval<const R&>())))> : std::true_type {}

namespace tachyon::math::internal {

template <typename Coefficients, typename SFINAE = void>
class MultivariatePolynomialOp;

SUPPORTS_POLY_OPERATOR(Add);
SUPPORTS_POLY_OPERATOR(Sub);

template <typename Evaluations, typename SFINAE = void>
class MultilinearExtensionOp;

SUPPORTS_MLE_OPERATOR(Add);
SUPPORTS_MLE_OPERATOR(Sub);
SUPPORTS_MLE_OPERATOR(Mul);
SUPPORTS_MLE_OPERATOR(Div);
SUPPORTS_MLE_OPERATOR(Mod);

}  // namespace tachyon::math::internal

#undef SUPPORTS_POLY_OPERATOR
#undef SUPPORTS_MLE_OPERATOR

#endif  // TACHYON_MATH_POLYNOMIALS_MULTIVARIATE_SUPPORT_POLY_OPERATORS_H_
