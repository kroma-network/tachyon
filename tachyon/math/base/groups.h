#ifndef TACHYON_MATH_BASE_GROUPS_H_
#define TACHYON_MATH_BASE_GROUPS_H_

#include "tachyon/math/base/semigroups.h"

namespace tachyon::math {
namespace internal {

SUPPORTS_BINARY_OPERATOR(Div);
SUPPORTS_BINARY_OPERATOR(Sub);
SUPPORTS_BINARY_OPERATOR(Mod);

}  // namespace internal

template <typename G>
class MultiplicativeGroup : public MultiplicativeSemigroup<G> {
 public:
  template <
      typename G2,
      std::enable_if_t<internal::SupportsMul<G, G2>::value ||
                       internal::SupportsMulInPlace<G, G2>::value ||
                       internal::SupportsDiv<G, G2>::value ||
                       internal::SupportsDivInPlace<G, G2>::value>* = nullptr>
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
class AdditiveGroup : public AdditiveSemigroup<G> {
 public:
  template <
      typename G2,
      std::enable_if_t<internal::SupportsAdd<G, G2>::value ||
                       internal::SupportsAddInPlace<G, G2>::value ||
                       internal::SupportsSub<G, G2>::value ||
                       internal::SupportsSubInPlace<G, G2>::value>* = nullptr>
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
    return ret.NegInPlace();
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_GROUPS_H_
