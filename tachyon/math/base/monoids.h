#ifndef TACHYON_MATH_BASE_MONOIDS_H_
#define TACHYON_MATH_BASE_MONOIDS_H_

#include "absl/types/span.h"

#include "tachyon/math/base/gmp_util.h"

namespace tachyon {
namespace math {
namespace internal {

template <typename L, typename R, typename = void>
struct SupportsMul : std::false_type {};

template <typename L, typename R>
struct SupportsMul<
    L, R, decltype(void(std::declval<L>().Mul(std::declval<const R&>())))>
    : std::true_type {};

template <typename L, typename R, typename = void>
struct SupportsMulInPlace : std::false_type {};

template <typename L, typename R>
struct SupportsMulInPlace<L, R,
                          decltype(void(std::declval<L>().MulInPlace(
                              std::declval<const R&>())))> : std::true_type {};

template <typename T, typename = void>
struct SupportsSquareInPlace : std::false_type {};

template <typename T>
struct SupportsSquareInPlace<T,
                             decltype(void(std::declval<T>().SquareInPlace()))>
    : std::true_type {};

template <typename L, typename R, typename = void>
struct SupportsAdd : std::false_type {};

template <typename L, typename R>
struct SupportsAdd<
    L, R, decltype(void(std::declval<L>().Add(std::declval<const R&>())))>
    : std::true_type {};

template <typename L, typename R, typename = void>
struct SupportsAddInPlace : std::false_type {};

template <typename L, typename R>
struct SupportsAddInPlace<L, R,
                          decltype(void(std::declval<L>().AddInPlace(
                              std::declval<const R&>())))> : std::true_type {};

template <typename T, typename = void>
struct SupportsDoubleInPlace : std::false_type {};

template <typename T>
struct SupportsDoubleInPlace<T,
                             decltype(void(std::declval<T>().DoubleInPlace()))>
    : std::true_type {};

}  // namespace internal

template <typename G>
class MultiplicativeMonoid {
 public:
  template <typename G2>
  constexpr auto operator*(const G2& other) const {
    if constexpr (internal::SupportsMul<G, G2>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Mul(other);
    } else {
      G g = *static_cast<const G*>(this);
      return g.MulInPlace(other);
    }
  }

  template <
      typename G2,
      std::enable_if_t<internal::SupportsMulInPlace<G, G2>::value>* = nullptr>
  constexpr auto& operator*=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->MulInPlace(other);
  }

  [[nodiscard]] constexpr auto Square() const {
    if constexpr (internal::SupportsSquareInPlace<G>::value) {
      G g = *static_cast<const G*>(this);
      return g.SquareInPlace();
    } else {
      return operator*(static_cast<const G&>(*this));
    }
  }

  [[nodiscard]] G Pow(const mpz_class& exponent) const {
    auto it = gmp::BitIteratorBE::begin(&exponent);
    auto end = gmp::BitIteratorBE::end(&exponent);
    const G& self = *static_cast<const G*>(this);
    G g = G::One();
    while (it != end) {
      if constexpr (internal::SupportsSquareInPlace<G>::value) {
        g.SquareInPlace();
      } else {
        g *= g;
      }

      if (*it) {
        g *= self;
      }
      ++it;
    }
    return g;
  }

  static G PowWithTable(absl::Span<G> powers_of_2, const mpz_class& exponent) {
    auto it = gmp::BitIteratorLE::begin(&exponent);
    auto end = gmp::BitIteratorLE::end(&exponent);
    G g = G::One();
    size_t i = 0;
    while (it != end) {
      if (*it) {
        g *= powers_of_2[i];
      }
      ++it;
      ++i;
    }
    return g;
  }
};

template <typename G>
class AdditiveMonoid {
 public:
  template <typename G2>
  constexpr auto operator+(const G2& other) const {
    if constexpr (internal::SupportsAdd<G, G2>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Add(other);
    } else {
      G g = *static_cast<const G*>(this);
      return g.AddInPlace(other);
    }
  }

  template <
      typename G2,
      std::enable_if_t<internal::SupportsAddInPlace<G, G2>::value>* = nullptr>
  constexpr auto& operator+=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->AddInPlace(other);
  }

  [[nodiscard]] constexpr auto Double() const {
    if constexpr (internal::SupportsDoubleInPlace<G>::value) {
      G g = *static_cast<const G*>(this);
      return g.DoubleInPlace();
    } else {
      return operator+(static_cast<const G&>(*this));
    }
  }

  template <
      typename T = G,
      std::enable_if_t<internal::SupportsDoubleInPlace<T>::value &&
                       internal::SupportsAddInPlace<T, G>::value>* = nullptr>
  auto operator*(const mpz_class& scalar) const {
    const G* g = static_cast<const G*>(this);
    G ret = G::Zero();
    auto it = gmp::BitIteratorBE::begin(&scalar);
    auto end = gmp::BitIteratorBE::end(&scalar);
    while (it != end) {
      ret.DoubleInPlace();
      if (*it) {
        ret += *g;
      }
      ++it;
    }
    return ret;
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_MONOIDS_H_
