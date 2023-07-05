#ifndef TACHYON_MATH_BASE_FIELD_H_
#define TACHYON_MATH_BASE_FIELD_H_

#include "tachyon/math/base/rings.h"

namespace tachyon {
namespace math {

template <typename F>
class Field : public AdditiveGroup<F>, public MultiplicativeGroup<F> {
 public:
  template <typename InputIterator,
            std::enable_if_t<std::is_same_v<
                F, base::internal::iterator_value_t<InputIterator>>>* = nullptr>
  constexpr static F SumOfProducts(InputIterator a_first, InputIterator a_last,
                                   InputIterator b_first,
                                   InputIterator b_last) {
    return Ring<F>::SumOfProducts(std::move(a_first), std::move(a_last),
                                  std::move(b_first), std::move(b_last));
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_FIELD_H_
