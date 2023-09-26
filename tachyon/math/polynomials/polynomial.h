#ifndef TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_
#define TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/base/rings.h"

namespace tachyon::math {

template <typename Derived>
class Polynomial : public Ring<Derived> {};

template <typename Derived>
class MultiplicativeIdentity<Polynomial<Derived>> {
 public:
  using P = Polynomial<Derived>;

  static const P& One() {
    static base::NoDestructor<P> one(P::One());
    return *one;
  }

  constexpr static bool IsOne(const P& value) { return value.IsOne(); }
};

template <typename Derived>
class AdditiveIdentity<Polynomial<Derived>> {
 public:
  using P = Polynomial<Derived>;

  static const P& Zero() {
    static base::NoDestructor<P> zero(P::Zero());
    return *zero;
  }

  constexpr static bool IsZero(const P& value) { return value.IsZero(); }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_
