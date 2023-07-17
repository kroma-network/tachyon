#ifndef TACHYON_MATH_BASE_MONOIDS_H_
#define TACHYON_MATH_BASE_MONOIDS_H_

#include "absl/types/span.h"

#include "tachyon/math/base/bit_iterator.h"

#define SUPPORTS_BINARY_OPERATOR(Name)                                        \
  template <typename L, typename R, typename = void>                          \
  struct Supports##Name : std::false_type {};                                 \
                                                                              \
  template <typename L, typename R>                                           \
  struct Supports##Name<                                                      \
      L, R, decltype(void(std::declval<L>().Name(std::declval<const R&>())))> \
      : std::true_type {};                                                    \
                                                                              \
  template <typename L, typename R, typename = void>                          \
  struct Supports##Name##InPlace : std::false_type {};                        \
                                                                              \
  template <typename L, typename R>                                           \
  struct Supports##Name##InPlace<L, R,                                        \
                                 decltype(void(                               \
                                     std::declval<L>().Name##InPlace(         \
                                         std::declval<const R&>())))>         \
      : std::true_type {}

#define SUPPORTS_UNARY_IN_PLACE_OPERATOR(Name)                               \
  template <typename T, typename = void>                                     \
  struct Supports##Name##InPlace : std::false_type {};                       \
                                                                             \
  template <typename T>                                                      \
  struct Supports##Name##InPlace<T, decltype(void(                           \
                                        std::declval<T>().Name##InPlace()))> \
      : std::true_type {}

namespace tachyon {
namespace math {
namespace internal {

SUPPORTS_BINARY_OPERATOR(Mul);
SUPPORTS_UNARY_IN_PLACE_OPERATOR(Square);
SUPPORTS_BINARY_OPERATOR(Add);
SUPPORTS_UNARY_IN_PLACE_OPERATOR(Double);

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

  template <size_t N>
  [[nodiscard]] G Pow(const BigInt<N>& exponent) const {
    auto it = BitIteratorBE<BigInt<N>>::begin(&exponent, true);
    auto end = BitIteratorBE<BigInt<N>>::end(&exponent);
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

  template <size_t N>
  static G PowWithTable(absl::Span<const G> powers_of_2,
                        const BigInt<N>& exponent) {
    auto it = BitIteratorLE<BigInt<N>>::begin(&exponent);
    auto end = BitIteratorLE<BigInt<N>>::end(&exponent, true);
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

  // FIXME(chokobole): In g++ (Ubuntu 11.3.0-1ubuntu1~22.04.1) 11.3.0, if I use
  // the function below, then it gives me an error "error: request for member
  // 'operator*' is ambiguous".
  // constexpr auto operator*(const BigInt<N>& scalar) const {
  template <size_t N>
  constexpr auto ScalarMul(const BigInt<N>& scalar) const {
    const G* g = static_cast<const G*>(this);
    G ret = G::Zero();
    auto it = BitIteratorBE<BigInt<N>>::begin(&scalar, true);
    auto end = BitIteratorBE<BigInt<N>>::end(&scalar);
    while (it != end) {
      if constexpr (internal::SupportsDoubleInPlace<G>::value) {
        ret.DoubleInPlace();
      } else {
        ret = ret.Double();
      }
      if (*it) {
        if constexpr (internal::SupportsAddInPlace<G, G>::value) {
          ret.AddInPlace(*g);
        } else {
          ret = ret.Add(*g);
        }
      }
      ++it;
    }
    return ret;
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_MONOIDS_H_
