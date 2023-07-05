#ifndef TACHYON_MATH_BASE_GROUPS_H_
#define TACHYON_MATH_BASE_GROUPS_H_

#include "tachyon/math/base/monoids.h"

namespace tachyon {
namespace math {
namespace internal {

template <typename T, typename = void>
struct SupportsDiv : std::false_type {};

template <typename T>
struct SupportsDiv<T, decltype(void(std::declval<T>().Div(
                          std::declval<const T&>())))> : std::true_type {};

template <typename T, typename = void>
struct SupportsSub : std::false_type {};

template <typename T>
struct SupportsSub<T, decltype(void(std::declval<T>().Sub(
                          std::declval<const T&>())))> : std::true_type {};

}  // namespace internal

template <typename G>
class MultiplicativeGroup : public MultiplicativeMonoid<G> {
 public:
  constexpr G operator/(const G& other) const {
    if constexpr (internal::SupportsDiv<G>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Div(other);
    } else {
      G g = *static_cast<const G*>(this);
      return g.DivInPlace(other);
    }
  }

  constexpr G& operator/=(const G& other) {
    G* g = static_cast<G*>(this);
    return g->DivInPlace(other);
  }

  [[nodiscard]] constexpr G Inverse() const {
    G ret = *static_cast<const G*>(this);
    return ret.InverseInPlace();
  }
};

template <typename G>
class AdditiveGroup : public AdditiveMonoid<G> {
 public:
  constexpr G operator-(const G& other) const {
    if constexpr (internal::SupportsSub<G>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Sub(other);
    } else {
      G g = *static_cast<const G*>(this);
      return g.SubInPlace(other);
    }
  }

  constexpr G& operator-=(const G& other) {
    G* g = static_cast<G*>(this);
    return g->SubInPlace(other);
  }

  constexpr G operator-() const {
    const G* g = static_cast<const G*>(this);
    return g->Negative();
  }

  [[nodiscard]] constexpr G Negative() const {
    G ret = *static_cast<const G*>(this);
    return ret.NegativeInPlace();
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_GROUPS_H_
