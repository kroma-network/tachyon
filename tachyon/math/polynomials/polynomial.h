#ifndef TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_
#define TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_

#include <type_traits>

#include "tachyon/math/base/ring.h"
#include "tachyon/math/polynomials/polynomial_traits_forward.h"

namespace tachyon::math {
template <typename T, typename SFINAE = void>
class Polynomial;

template <typename Derived>
class Polynomial<
    Derived, std::enable_if_t<PolynomialTraits<Derived>::kIsCoefficientForm>>
    : public Ring<Derived> {};

template <typename Derived>
class Polynomial<Derived,
                 std::enable_if_t<PolynomialTraits<Derived>::kIsEvaluationForm>>
    : public Ring<Derived> {};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_
