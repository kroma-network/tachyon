#ifndef TACHYON_MATH_BASE_MONOIDS_H_
#define TACHYON_MATH_BASE_MONOIDS_H_

#include "absl/types/span.h"

#include "tachyon/math/base/gmp_util.h"

namespace tachyon {
namespace math {
namespace internal {

template <typename T, typename = void>
struct SupportsMul : std::false_type {};

template <typename T>
struct SupportsMul<T, decltype(void(std::declval<T>().Mul(
                          std::declval<const T&>())))> : std::true_type {};

template <typename T, typename = void>
struct SupportsSquareInPlace : std::false_type {};

template <typename T>
struct SupportsSquareInPlace<T,
                             decltype(void(std::declval<T>().SquareInPlace()))>
    : std::true_type {};

template <typename T, typename = void>
struct SupportsAdd : std::false_type {};

template <typename T>
struct SupportsAdd<T, decltype(void(std::declval<T>().Add(
                          std::declval<const T&>())))> : std::true_type {};

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
  constexpr G operator*(const G& other) const {
    if constexpr (internal::SupportsMul<G>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Mul(other);
    } else {
      G g = *static_cast<const G*>(this);
      return g.MulInPlace(other);
    }
  }

  constexpr G& operator*=(const G& other) {
    G* g = static_cast<G*>(this);
    return g->MulInPlace(other);
  }

  [[nodiscard]] constexpr G Square() const {
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
  constexpr G operator+(const G& other) const {
    if constexpr (internal::SupportsAdd<G>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Add(other);
    } else {
      G g = *static_cast<const G*>(this);
      return g.AddInPlace(other);
    }
  }

  constexpr G& operator+=(const G& other) {
    G* g = static_cast<G*>(this);
    return g->AddInPlace(other);
  }

  [[nodiscard]] constexpr G Double() const {
    if constexpr (internal::SupportsDoubleInPlace<G>::value) {
      G g = *static_cast<const G*>(this);
      return g.DoubleInPlace();
    } else {
      return operator+(static_cast<const G&>(*this));
    }
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_MONOIDS_H_
