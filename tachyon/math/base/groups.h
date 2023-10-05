#ifndef TACHYON_MATH_BASE_GROUPS_H_
#define TACHYON_MATH_BASE_GROUPS_H_

#include "tachyon/math/base/semigroups.h"

namespace tachyon::math {
namespace internal {

SUPPORTS_BINARY_OPERATOR(Div);
SUPPORTS_BINARY_OPERATOR(Sub);
SUPPORTS_BINARY_OPERATOR(Mod);

}  // namespace internal

// Group 'G' is a set of elements together with a binary operation (called the
// group operation) that together satisfy the four fundamental properties of
// closure, associative, the identity property, and the inverse property.
// See https://mathworld.wolfram.com/Group.html

// MultiplicativeGroup is a group with the group operation '*'.
// MultiplicativeGroup supports division and inversion, inheriting the
// properties of MultiplicativeSemigroup.
template <typename G>
class MultiplicativeGroup : public MultiplicativeSemigroup<G> {
 public:
  // Division:
  //   1) a / b if division is supported.
  //   2) a * b⁻¹ otherwise
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

  // Division in place: a /= b
  template <
      typename G2,
      std::enable_if_t<internal::SupportsDivInPlace<G, G2>::value>* = nullptr>
  constexpr auto& operator/=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->DivInPlace(other);
  }

  // Division in place: a *= b⁻¹
  template <
      typename G2,
      std::enable_if_t<!internal::SupportsDivInPlace<G, G2>::value &&
                       internal::SupportsMulInPlace<G, G2>::value>* = nullptr>
  constexpr auto& operator/=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->MulInPlace(other.Inverse());
  }

  // Inverse: a⁻¹
  [[nodiscard]] constexpr auto Inverse() const {
    G ret = *static_cast<const G*>(this);
    return ret.InverseInPlace();
  }
};
// AdditiveGroup is a group with the group operation '+'.
// AdditiveGroup supports subtraction and negation, inheriting the
// properties of AdditiveSemigroup.
template <typename G>
class AdditiveGroup : public AdditiveSemigroup<G> {
 public:
  // Subtraction:
  //   1) a - b if subtraction is supported.
  //   2) a + (-b) otherwise
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

  // Subtraction in place: a -= b
  template <
      typename G2,
      std::enable_if_t<internal::SupportsSubInPlace<G, G2>::value>* = nullptr>
  constexpr auto& operator-=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->SubInPlace(other);
  }

  // Subtraction in place: a += (-b)
  template <
      typename G2,
      std::enable_if_t<!internal::SupportsSubInPlace<G, G2>::value &&
                       internal::SupportsAddInPlace<G, G2>::value>* = nullptr>
  constexpr auto& operator-=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->AddInPlace(other.Negative());
  }

  // Negation: -a
  constexpr auto operator-() const { return Negative(); }

  [[nodiscard]] constexpr auto Negative() const {
    G ret = *static_cast<const G*>(this);
    return ret.NegInPlace();
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_GROUPS_H_
