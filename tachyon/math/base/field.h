#ifndef TACHYON_MATH_BASE_FIELD_H_
#define TACHYON_MATH_BASE_FIELD_H_

#include <utility>

#include "tachyon/math/base/rings.h"

namespace tachyon::math {

// Field is any set of elements that satisfies the field axioms for both
// addition and multiplication and is commutative division algebra.
// Simply put, a field is a ring in which multiplicative commutativity exists,
// and every non-zero element has a multiplicative inverse.
// See https://mathworld.wolfram.com/Field.html

// The Field supports SumOfProducts and BatchInverse, inheriting the properties
// of both AdditiveGroup and MultiplicativeGroup.
template <typename F>
class Field : public AdditiveGroup<F>, public MultiplicativeGroup<F> {
 public:
  // Sum of products: a₁ * b₁ + a₂ * b₂ + ... + aₙ * bₙ
  template <typename Container>
  constexpr static F SumOfProducts(const Container& a, const Container& b) {
    return Ring<F>::SumOfProducts(a, b);
  }

  // Sum of products: a₁ * b₁ + a₂ * b₂ + ... + aₙ * bₙ
  template <typename Container>
  constexpr static F SumOfProductsSerial(const Container& a,
                                         const Container& b) {
    return Ring<F>::SumOfProductsSerial(a, b);
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_FIELD_H_
