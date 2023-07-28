#ifndef TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_
#define TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_

#include <stddef.h>

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/base/rings.h"

namespace tachyon::math {

template <typename T, typename SFINAE = void>
class CoefficientsTraits;

template <typename Derived>
class Polynomial : public Ring<Derived> {
 public:
  constexpr static const size_t kMaxDegree = Derived::kMaxDegree;

  using Field = typename CoefficientsTraits<Derived>::Field;

  constexpr size_t Degree() const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoDegree();
  }

  constexpr Field Evaluate(const Field& point) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoEvaluate(point);
  }
};

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
