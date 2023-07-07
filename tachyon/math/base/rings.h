#ifndef TACHYON_MATH_BASE_RINGS_H_
#define TACHYON_MATH_BASE_RINGS_H_

#include <type_traits>

#include "tachyon/base/template_util.h"
#include "tachyon/math/base/groups.h"

namespace tachyon {
namespace math {

template <typename F>
class Ring : public AdditiveGroup<F>, public MultiplicativeMonoid<F> {
 public:
  template <
      typename InputIterator,
      std::enable_if_t<std::is_same_v<F, base::iter_value_t<InputIterator>>>* =
          nullptr>
  constexpr static F SumOfProducts(InputIterator a_first, InputIterator a_last,
                                   InputIterator b_first,
                                   InputIterator b_last) {
    F sum = F::Zero();
    while (a_first != a_last) {
      sum += (*a_first * *b_first);
      ++a_first;
      ++b_first;
    }
    return sum;
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_RINGS_H_
