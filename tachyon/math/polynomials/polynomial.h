#ifndef TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_
#define TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_

#include <stddef.h>

#include "tachyon/math/base/rings.h"

namespace tachyon {
namespace math {

template <typename T, typename SFINAE = void>
class CoefficientsTraits;

template <typename Derived>
class Polynomial : public Ring<Derived> {
 public:
  constexpr static const size_t MAX_DEGREE = Derived::MAX_DEGREE;

  using Field = typename CoefficientsTraits<Derived>::Field;

  constexpr size_t Degree() const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoDegree();
  }

  constexpr Field Evaluate(const Field& point) const {
    const Derived* derived = static_cast<const Derived*>(this);
    return derived->DoEvaluate(point);
  }

  // AdditiveGroup methods
  Derived& NegativeInPlace() {
    NOTIMPLEMENTED();
    return static_cast<Derived&>(*this);
  }

 private:
  friend class AdditiveMonoid<Derived>;
  friend class AdditiveGroup<Derived>;
  friend class MultiplicativeMonoid<Derived>;

  // AdditiveMonoid methods
  constexpr Derived& AddInPlace(const Derived& other) {
    NOTIMPLEMENTED();
    return *this;
  }

  // AdditiveGroup methods
  constexpr Derived& SubInPlace(const Derived& other) {
    NOTIMPLEMENTED();
    return *this;
  }

  // MultiplicativeMonoid methods
  constexpr Derived& MulInPlace(const Derived& other) {
    NOTIMPLEMENTED();
    return *this;
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_
