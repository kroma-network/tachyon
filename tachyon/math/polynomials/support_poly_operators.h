#ifndef TACHYON_MATH_POLYNOMIALS_SUPPORT_POLY_OPERATORS_H_
#define TACHYON_MATH_POLYNOMIALS_SUPPORT_POLY_OPERATORS_H_

#define SUPPORTS_POLY_OPERATOR(Name)                             \
  template <typename T, typename L, typename R, typename = void> \
  struct SupportsPoly##Name : std::false_type {};                \
                                                                 \
  template <typename T, typename L, typename R, typename = void> \
  struct SupportsPoly##Name##InPlace : std::false_type {}

namespace tachyon::math::internal {

SUPPORTS_POLY_OPERATOR(Add);
SUPPORTS_POLY_OPERATOR(Sub);
SUPPORTS_POLY_OPERATOR(Mul);
SUPPORTS_POLY_OPERATOR(Div);
SUPPORTS_POLY_OPERATOR(Mod);

}  // namespace tachyon::math::internal

#undef SUPPORTS_POLY_OPERATOR

#endif  // TACHYON_MATH_POLYNOMIALS_SUPPORT_POLY_OPERATORS_H_
