#ifndef TACHYON_MATH_BASE_GROUPS_H_
#define TACHYON_MATH_BASE_GROUPS_H_

#include "tachyon/math/base/monoids.h"

namespace tachyon {
namespace math {
namespace internal {

SUPPORTS_BINARY_OPERATOR(Div);
SUPPORTS_BINARY_OPERATOR(Sub);

}  // namespace internal

template <typename G>
class MultiplicativeGroup : public MultiplicativeMonoid<G> {
 public:
  template <typename G2>
  constexpr auto operator/(const G2& other) const {
    if constexpr (internal::SupportsDiv<G, G2>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Div(other);
    } else if constexpr (internal::SupportsDivInPlace<G, G2>::value) {
      G g = *static_cast<const G*>(this);
      return g.DivInPlace(other);
    } else {
      return this->operator*(other.Inverse());
    }
  }

  template <
      typename G2,
      std::enable_if_t<internal::SupportsDivInPlace<G, G2>::value>* = nullptr>
  constexpr auto& operator/=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->DivInPlace(other);
  }

  template <
      typename G2,
      std::enable_if_t<!internal::SupportsDivInPlace<G, G2>::value &&
                       internal::SupportsMulInPlace<G, G2>::value>* = nullptr>
  constexpr auto& operator/=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->MulInPlace(other.Inverse());
  }

  [[nodiscard]] constexpr auto Inverse() const {
    G ret = *static_cast<const G*>(this);
    return ret.InverseInPlace();
  }
};

template <typename G>
class AdditiveGroup : public AdditiveMonoid<G> {
 public:
  template <typename G2>
  constexpr auto operator-(const G2& other) const {
    if constexpr (internal::SupportsSub<G, G2>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Sub(other);
    } else if constexpr (internal::SupportsSubInPlace<G, G2>::value) {
      G g = *static_cast<const G*>(this);
      return g.SubInPlace(other);
    } else {
      return this->operator+(other.Negative());
    }
  }

  template <
      typename G2,
      std::enable_if_t<internal::SupportsSubInPlace<G, G2>::value>* = nullptr>
  constexpr auto& operator-=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->SubInPlace(other);
  }

  template <
      typename G2,
      std::enable_if_t<!internal::SupportsSubInPlace<G, G2>::value &&
                       internal::SupportsAddInPlace<G, G2>::value>* = nullptr>
  constexpr auto& operator-=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->AddInPlace(other.Negative());
  }

  constexpr auto operator-() const { return Negative(); }

  [[nodiscard]] constexpr auto Negative() const {
    G ret = *static_cast<const G*>(this);
    return ret.NegativeInPlace();
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_GROUPS_H_
