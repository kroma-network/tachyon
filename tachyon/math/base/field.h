#ifndef TACHYON_MATH_BASE_FIELD_H_
#define TACHYON_MATH_BASE_FIELD_H_

#include <utility>

#include "tachyon/math/base/rings.h"

namespace tachyon::math {

// Field is any set of elements that satisfies the field axioms for both
// addition and multiplication and is commutative division algebra
// Simply put, a field is a ring in which multiplicative commutativity exists,
// and every non-zero element has a multiplicative inverse.
// See https://mathworld.wolfram.com/Field.html

// The Field supports SumOfProducts, inheriting the properties of both
// AdditiveGroup and MultiplicativeGroup.
template <typename F>
class Field : public AdditiveGroup<F>, public MultiplicativeGroup<F> {
 public:
  // Sum of products: a₁ * b₁ + a₂ * b₂ + ... + aₙ * bₙ
  template <
      typename InputIterator,
      std::enable_if_t<std::is_same_v<F, base::iter_value_t<InputIterator>>>* =
          nullptr>
  constexpr static F SumOfProducts(InputIterator a_first, InputIterator a_last,
                                   InputIterator b_first,
                                   InputIterator b_last) {
    return Ring<F>::SumOfProducts(std::move(a_first), std::move(a_last),
                                  std::move(b_first), std::move(b_last));
  }

  template <typename Container>
  constexpr static F SumOfProducts(const Container& a, const Container& b) {
    return Ring<F>::SumOfProducts(a, b);
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_FIELD_H_
