#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_FORWARD_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_FORWARD_H_

#define SUPPORTS_POLY_OPERATOR(Name)                                        \
  template <typename Coefficients, typename L, typename R, typename = void> \
  struct SupportsPoly##Name : std::false_type {};                           \
                                                                            \
  template <typename Coefficients, typename L, typename R>                  \
  struct SupportsPoly##Name<                                                \
      Coefficients, L, R,                                                   \
      decltype(void(UnivariatePolynomialOp<Coefficients>::Name(             \
          std::declval<const L&>(), std::declval<const R&>())))>            \
      : std::true_type {};                                                  \
                                                                            \
  template <typename Coefficients, typename L, typename R, typename = void> \
  struct SupportsPoly##Name##InPlace : std::false_type {};                  \
                                                                            \
  template <typename Coefficients, typename L, typename R>                  \
  struct SupportsPoly##Name##InPlace<                                       \
      Coefficients, L, R,                                                   \
      decltype(void(UnivariatePolynomialOp<Coefficients>::Name##InPlace(    \
          std::declval<L&>(), std::declval<const R&>())))> : std::true_type {}

namespace tachyon {
namespace math {
namespace internal {

template <typename Coefficients, typename SFINAE = void>
class UnivariatePolynomialOp;

SUPPORTS_POLY_OPERATOR(Add);
SUPPORTS_POLY_OPERATOR(Sub);
SUPPORTS_POLY_OPERATOR(Mul);

}  // namespace internal
}  // namespace math
}  // namespace tachyon

#undef SUPPORTS_POLY_OPERATOR

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_POLYNOMIAL_OPS_FORWARD_H_
