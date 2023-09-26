#ifndef TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_
#define TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_

#include "tachyon/math/base/rings.h"

namespace tachyon::math {

template <typename Derived>
class Polynomial : public Ring<Derived> {};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_
