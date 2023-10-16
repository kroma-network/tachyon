#ifndef TACHYON_MATH_BASE_GROUPS_H_
#define TACHYON_MATH_BASE_GROUPS_H_

#include "tachyon/base/types/always_false.h"
#include "tachyon/math/base/semigroups.h"

namespace tachyon::math {
namespace internal {

SUPPORTS_BINARY_OPERATOR(Div);
SUPPORTS_BINARY_OPERATOR(Mod);
SUPPORTS_UNARY_IN_PLACE_OPERATOR(Inverse);
SUPPORTS_BINARY_OPERATOR(Sub);
SUPPORTS_UNARY_IN_PLACE_OPERATOR(Neg);

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
  //   1) a /= b if division is supported.
  //   2) a *= b⁻¹ otherwise
  template <
      typename G2,
      std::enable_if_t<internal::SupportsDivInPlace<G, G2>::value ||
                       internal::SupportsMulInPlace<G, G2>::value>* = nullptr>
  constexpr G& operator/=(const G2& other) {
    if constexpr (internal::SupportsDivInPlace<G, G2>::value) {
      G* g = static_cast<G*>(this);
      return g->DivInPlace(other);
    } else if constexpr (internal::SupportsMulInPlace<G, G2>::value) {
      G* g = static_cast<G*>(this);
      return g->MulInPlace(other.Inverse());
    } else {
      static_assert(base::AlwaysFalse<G>);
    }
  }

  // Inverse: a⁻¹
  template <
      typename G2 = G,
      std::enable_if_t<internal::SupportsInverseInPlace<G2>::value>* = nullptr>
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

  // Subtraction in place:
  //   1) a -= b if subtraction is supported.
  //   2) a += (-b) otherwise
  template <
      typename G2,
      std::enable_if_t<internal::SupportsSubInPlace<G, G2>::value ||
                       internal::SupportsAddInPlace<G, G2>::value>* = nullptr>
  constexpr G& operator-=(const G2& other) {
    if constexpr (internal::SupportsSubInPlace<G, G2>::value) {
      G* g = static_cast<G*>(this);
      return g->SubInPlace(other);
    } else if constexpr (internal::SupportsAddInPlace<G, G2>::value) {
      G* g = static_cast<G*>(this);
      return g->AddInPlace(other.Negative());
    } else {
      static_assert(base::AlwaysFalse<G>);
    }
  }

  // Negation: -a
  constexpr auto operator-() const {
    if constexpr (internal::SupportsNegInPlace<G>::value) {
      G g = *static_cast<const G*>(this);
      return g.NegInPlace();
    } else {
      const G* g = static_cast<const G*>(this);
      return g->Negative();
    }
  }

  template <
      typename G2 = G,
      std::enable_if_t<internal::SupportsNegInPlace<G2>::value>* = nullptr>
  [[nodiscard]] constexpr auto Negative() const {
    G ret = *static_cast<const G*>(this);
    return ret.NegInPlace();
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_GROUPS_H_
