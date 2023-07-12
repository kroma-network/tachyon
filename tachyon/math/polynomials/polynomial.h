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

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_POLYNOMIALS_POLYNOMIAL_H_
