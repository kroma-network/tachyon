#ifndef TACHYON_MATH_BASE_SEMIGROUPS_H_
#define TACHYON_MATH_BASE_SEMIGROUPS_H_

#include "absl/types/span.h"

#include "tachyon/math/base/big_int.h"
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

namespace tachyon::math {
namespace internal {

SUPPORTS_BINARY_OPERATOR(Mul);
SUPPORTS_UNARY_IN_PLACE_OPERATOR(Square);
SUPPORTS_BINARY_OPERATOR(Add);
SUPPORTS_UNARY_IN_PLACE_OPERATOR(Double);

template <typename G>
struct MultiplicativeSemigroupTraits {
  using ReturnTy = G;
};

template <typename G>
struct AdditiveSemigroupTraits {
  using ReturnTy = G;
};

}  // namespace internal

// Semigroup <S, ○> is a mathematical object defined for a set 'S' and a binary
// operator '○' in which the operation is associative.
// Associativity: (a ○ b) ○ c = a ○ (b ○ c)
// See https://mathworld.wolfram.com/Semigroup.html

// MultiplicativeSemigroup is a semigroup with a multiplicative operator.
template <typename G>
class MultiplicativeSemigroup {
 public:
  template <
      typename G2,
      std::enable_if_t<internal::SupportsMul<G, G2>::value ||
                       internal::SupportsMulInPlace<G, G2>::value>* = nullptr>

  // Multiplication: a * b
  constexpr auto operator*(const G2& other) const {
    if constexpr (internal::SupportsMul<G, G2>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Mul(other);
    } else {
      G g = *static_cast<const G*>(this);
      return g.MulInPlace(other);
    }
  }

  // Multiplication in place: a *= b
  template <
      typename G2,
      std::enable_if_t<internal::SupportsMulInPlace<G, G2>::value>* = nullptr>
  constexpr G& operator*=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->MulInPlace(other);
  }

  // a.Square(): a²
  [[nodiscard]] constexpr auto Square() const {
    if constexpr (internal::SupportsSquareInPlace<G>::value) {
      G g = *static_cast<const G*>(this);
      return g.SquareInPlace();
    } else {
      return operator*(static_cast<const G&>(*this));
    }
  }

  // a.Pow(e): aᵉ
  // Square it as much as possible and multiply the remainder.
  // ex) a¹³ = (((a)² * a)²)² * a
  template <size_t N,
            typename ReturnTy =
                typename internal::MultiplicativeSemigroupTraits<G>::ReturnTy>
  [[nodiscard]] constexpr ReturnTy Pow(const BigInt<N>& exponent) const {
    const G* g = static_cast<const G*>(this);
    ReturnTy ret = ReturnTy::One();
    auto it = BitIteratorBE<BigInt<N>>::begin(&exponent, true);
    auto end = BitIteratorBE<BigInt<N>>::end(&exponent);
    while (it != end) {
      if constexpr (internal::SupportsSquareInPlace<G>::value) {
        ret.SquareInPlace();
      } else {
        ret = ret.Square();
      }
      if (*it) {
        if constexpr (internal::SupportsMulInPlace<ReturnTy, G>::value) {
          ret.MulInPlace(*g);
        } else {
          ret = ret.Mul(*g);
        }
      }
      ++it;
    }
    return ret;
  }

  // Computes the power of a base element using a pre-computed table of powers
  // of two, instead of performing repeated multiplications.
  template <size_t N,
            typename ReturnTy =
                typename internal::MultiplicativeSemigroupTraits<G>::ReturnTy>
  static ReturnTy PowWithTable(absl::Span<const G> powers_of_2,
                               const BigInt<N>& exponent) {
    auto it = BitIteratorLE<BigInt<N>>::begin(&exponent);
    auto end = BitIteratorLE<BigInt<N>>::end(&exponent, true);
    ReturnTy g = ReturnTy::One();
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

// AdditiveSemigroup is a semigroup with an additive operator.
template <typename G>
class AdditiveSemigroup {
 public:
  // Addition: a + b
  template <
      typename G2,
      std::enable_if_t<internal::SupportsAdd<G, G2>::value ||
                       internal::SupportsAddInPlace<G, G2>::value>* = nullptr>
  constexpr auto operator+(const G2& other) const {
    if constexpr (internal::SupportsAdd<G, G2>::value) {
      const G* g = static_cast<const G*>(this);
      return g->Add(other);
    } else {
      G g = *static_cast<const G*>(this);
      return g.AddInPlace(other);
    }
  }

  // Addition in place: a += b
  template <
      typename G2,
      std::enable_if_t<internal::SupportsAddInPlace<G, G2>::value>* = nullptr>
  constexpr G& operator+=(const G2& other) {
    G* g = static_cast<G*>(this);
    return g->AddInPlace(other);
  }

  // a.Double(): 2a
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
  template <size_t N,
            typename ReturnTy =
                typename internal::AdditiveSemigroupTraits<G>::ReturnTy>
  [[nodiscard]] constexpr ReturnTy ScalarMul(const BigInt<N>& scalar) const {
    const G* g = static_cast<const G*>(this);
    ReturnTy ret = ReturnTy::Zero();
    auto it = BitIteratorBE<BigInt<N>>::begin(&scalar, true);
    auto end = BitIteratorBE<BigInt<N>>::end(&scalar);
    while (it != end) {
      if constexpr (internal::SupportsDoubleInPlace<G>::value) {
        ret.DoubleInPlace();
      } else {
        ret = ret.Double();
      }
      if (*it) {
        if constexpr (internal::SupportsAddInPlace<ReturnTy, G>::value) {
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

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_SEMIGROUPS_H_
