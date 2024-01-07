#ifndef TACHYON_MATH_BASE_SEMIGROUPS_H_
#define TACHYON_MATH_BASE_SEMIGROUPS_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/base/types/always_false.h"
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

template <typename T, typename = void>
struct SupportsToBigInt : std::false_type {};

template <typename T>
struct SupportsToBigInt<T, decltype(void(std::declval<T>().ToBigInt()))>
    : std::true_type {};

template <typename T, typename = void>
struct SupportsSize : std::false_type {};

template <typename T>
struct SupportsSize<T, decltype(void(std::size(std::declval<T>())))>
    : std::true_type {};

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
  using ReturnTy =
      typename internal::MultiplicativeSemigroupTraits<G>::ReturnTy;

  // Multiplication: a * b
  template <
      typename G2,
      std::enable_if_t<internal::SupportsMul<G, G2>::value ||
                       internal::SupportsMulInPlace<G, G2>::value>* = nullptr>
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
  template <size_t N>
  [[nodiscard]] constexpr auto Pow(const BigInt<N>& exponent) const {
    return DoPow(exponent);
  }

  template <typename Scalar>
  [[nodiscard]] constexpr auto Pow(const Scalar& scalar) const {
    if constexpr (std::is_constructible_v<BigInt<1>, Scalar>) {
      return DoPow(BigInt<1>(scalar));
    } else if constexpr (internal::SupportsToBigInt<Scalar>::value) {
      return DoPow(scalar.ToBigInt());
    } else {
      static_assert(base::AlwaysFalse<G>);
    }
  }

  // Computes the power of a base element using a pre-computed table of powers
  // of two, instead of performing repeated multiplications.
  template <size_t N>
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

  // generator: g
  // return: [c, c * g, c * g², ..., c * g^{|size| - 1}]
  constexpr static std::vector<ReturnTy> GetSuccessivePowers(
      size_t size, const G& generator, const G& c = G::One()) {
    std::vector<ReturnTy> ret;
    ret.resize(size);
    size_t num_elems_per_thread =
        base::GetNumElementsPerThread(ret, kDefaultParallelThreshold);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < size; i += num_elems_per_thread) {
      ReturnTy pow = c * generator.Pow(i);
      for (size_t j = i; j < i + num_elems_per_thread && j < size; ++j) {
        ret[j] = pow;
        pow *= generator;
      }
    }
    return ret;
  }

 private:
  constexpr static size_t kDefaultParallelThreshold = 1024;

  template <size_t N>
  [[nodiscard]] constexpr ReturnTy DoPow(const BigInt<N>& exponent) const {
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
};

// AdditiveSemigroup is a semigroup with an additive operator.
template <typename G>
class AdditiveSemigroup {
 public:
  using ReturnTy = typename internal::AdditiveSemigroupTraits<G>::ReturnTy;

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
  template <size_t N>
  [[nodiscard]] constexpr auto ScalarMul(const BigInt<N>& scalar) const {
    return DoScalarMul(scalar);
  }

  template <typename Scalar>
  [[nodiscard]] constexpr auto ScalarMul(const Scalar& scalar) const {
    if constexpr (std::is_constructible_v<BigInt<1>, Scalar>) {
      return DoScalarMul(BigInt<1>(scalar));
    } else if constexpr (internal::SupportsToBigInt<Scalar>::value) {
      return DoScalarMul(scalar.ToBigInt());
    } else {
      static_assert(base::AlwaysFalse<G>);
    }
  }

  // Return false if the size of function arguments are not matched.
  // This supports 3 cases.
  //
  //   - Multi Scalar Multi Base
  //     scalars: [s₀, s₁, ..., sₙ₋₁]
  //     bases: [G₀, G₁, ..., Gₙ₋₁]
  //     outputs: [s₀G₀, s₁G₁, ..., sₙ₋₁Gₙ₋₁]
  //
  //   - Multi Scalar Single Base
  //     scalars: [s₀, s₁, ..., sₙ₋₁]
  //     base: G
  //     outputs: [s₀G, s₁G, ..., sₙ₋₁G]
  //
  //   - Single Scalar Multi Base
  //     scalar: s
  //     bases: [G₀, G₁, ..., Gₙ₋₁]
  //     outputs: [sG₀, sG₁, ..., sGₙ₋₁]
  template <typename ScalarOrScalars, typename BaseOrBases,
            typename OutputContainer>
  [[nodiscard]] constexpr static bool MultiScalarMul(
      const ScalarOrScalars& scalar_or_scalars,
      const BaseOrBases& base_or_bases, OutputContainer* outputs) {
    if constexpr (internal::SupportsSize<ScalarOrScalars>::value &&
                  internal::SupportsSize<BaseOrBases>::value) {
      return MultiScalarMulMSMB(scalar_or_scalars, base_or_bases, outputs);
    } else if constexpr (internal::SupportsSize<ScalarOrScalars>::value) {
      return MultiScalarMulMSSB(scalar_or_scalars, base_or_bases, outputs);
    } else if constexpr (internal::SupportsSize<BaseOrBases>::value) {
      return MultiScalarMulSSMB(scalar_or_scalars, base_or_bases, outputs);
    } else {
      static_assert(base::AlwaysFalse<G>);
    }
  }

  // generator: G
  // return: [0G, 1G, 2G, ..., (|size| - 1)G]
  // NOTE(chokobole): Unlike |GetSuccessivePowers()|, this doesn't have an
  // additional |c| parameter because I think there's no usecase that depends on
  // it.
  constexpr static std::vector<ReturnTy> GetSuccessiveScalarMuls(
      size_t size, const G& generator) {
    std::vector<ReturnTy> ret;
    ret.resize(size);
    size_t num_elems_per_thread =
        base::GetNumElementsPerThread(ret, kDefaultParallelThreshold);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < size; i += num_elems_per_thread) {
      ReturnTy scalar_mul = generator.ScalarMul(i);
      for (size_t j = i; j < i + num_elems_per_thread && j < size; ++j) {
        ret[j] = scalar_mul;
        scalar_mul += generator;
      }
    }
    return ret;
  }

  // Linear combination
  // - forward: a₀ * rⁿ⁻¹ + a₁ * rⁿ⁻² + ... + aₙ₋₁
  // - backward: a₀ + a₁ * r + ... + aₙ₋₁ * rⁿ⁻¹
  // NOTE(chokobole): For performance reasons, I recommend using
  // |LinearCombinationInPlace()| if possible.
  template <bool Forward, typename Container, typename T,
            typename ReturnTy =
                typename internal::AdditiveSemigroupTraits<G>::ReturnTy>
  constexpr static ReturnTy LinearCombination(const Container& values,
                                              const T& r) {
    size_t size = std::size(values);
    ReturnTy ret = ReturnTy::Zero();
    if constexpr (Forward) {
      for (size_t i = 0; i < size; ++i) {
        ret *= r;
        ret += values[i];
      }
    } else {
      for (size_t i = size - 1; i != SIZE_MAX; --i) {
        ret *= r;
        ret += values[i];
      }
    }
    return ret;
  }

  // Linear combination
  // - forward: a₀ * rⁿ⁻¹ + a₁ * rⁿ⁻² + ... + aₙ₋₁
  // - backward: a₀ + a₁ * r + ... + aₙ₋₁ * rⁿ⁻¹
  // NOTE(chokobole): This gives more performant result than
  // |LinearCombination()|.
  //
  // When using |LinearCombination()|, you can linearize groups as follows:
  //
  //   const std::vector<G> groups = {...};
  //   G ret = G::LinearCombination(groups, G::Random());
  //
  // When using |LinearCombinationInPlace()|, you can save additional allocation
  // cost.
  //
  //   // Note that |groups| are going to be changed.
  //   std::vector<G> groups = {...};
  //   G& ret = G::LinearCombinationInPlace(groups, G::Random());
  template <bool Forward, typename Container, typename T,
            typename ReturnTy =
                typename internal::AdditiveSemigroupTraits<G>::ReturnTy,
            std::enable_if_t<std::is_same_v<G, ReturnTy>>* = nullptr>
  constexpr static G& LinearCombinationInPlace(Container& values, const T& r) {
    size_t size = std::size(values);
    CHECK_GT(size, size_t{0});
    if constexpr (Forward) {
      G& ret = values[0];
      if (size > 1) {
        for (size_t i = 1; i < size; ++i) {
          ret *= r;
          ret += values[i];
        }
      }
      return ret;
    } else {
      G& ret = values[size - 1];
      if (size > 1) {
        for (size_t i = size - 2; i != SIZE_MAX; --i) {
          ret *= r;
          ret += values[i];
        }
      }
      return ret;
    }
  }

 private:
  constexpr static size_t kDefaultParallelThreshold = 1024;

  template <size_t N>
  [[nodiscard]] constexpr ReturnTy DoScalarMul(const BigInt<N>& scalar) const {
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

  // Multi Scalar Multi Base
  template <typename ScalarContainer, typename BaseContainer,
            typename OutputContainer>
  constexpr static bool MultiScalarMulMSMB(const ScalarContainer& scalars,
                                           const BaseContainer& bases,
                                           OutputContainer* outputs) {
    size_t size = scalars.size();
    if (size != std::size(bases)) return false;
    if (size != std::size(*outputs)) return false;
    if (size == 0) {
      LOG(ERROR) << "scalars and bases are empty";
      return false;
    }
    size_t num_elems_per_thread =
        base::GetNumElementsPerThread(scalars, kDefaultParallelThreshold);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < size; i += num_elems_per_thread) {
      for (size_t j = i; j < i + num_elems_per_thread && j < size; ++j) {
        (*outputs)[j] = bases[j].ScalarMul(scalars[j]);
      }
    }
    return true;
  }

  // Multi Scalar Single Base
  template <typename ScalarContainer, typename OutputContainer>
  constexpr static bool MultiScalarMulMSSB(const ScalarContainer& scalars,
                                           const G& base,
                                           OutputContainer* outputs) {
    size_t size = std::size(scalars);
    if (size != std::size(*outputs)) return false;
    if (size == 0) {
      LOG(ERROR) << "scalars are empty";
      return false;
    }
    size_t num_elems_per_thread =
        base::GetNumElementsPerThread(scalars, kDefaultParallelThreshold);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < size; i += num_elems_per_thread) {
      for (size_t j = i; j < i + num_elems_per_thread && j < size; ++j) {
        (*outputs)[j] = base.ScalarMul(scalars[j]);
      }
    }
    return true;
  }

  // Single Scalar Multi Base
  template <typename Scalar, typename BaseContainer, typename OutputContainer>
  constexpr static bool MultiScalarMulSSMB(const Scalar& scalar,
                                           const BaseContainer& bases,
                                           OutputContainer* outputs) {
    size_t size = std::size(bases);
    if (size != std::size(*outputs)) return false;
    if (size == 0) {
      LOG(ERROR) << "bases are empty";
      return false;
    }
    size_t num_elems_per_thread =
        base::GetNumElementsPerThread(bases, kDefaultParallelThreshold);
    OPENMP_PARALLEL_FOR(size_t i = 0; i < size; i += num_elems_per_thread) {
      for (size_t j = i; j < i + num_elems_per_thread && j < size; ++j) {
        (*outputs)[j] = bases[j].ScalarMul(scalar);
      }
    }
    return true;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_SEMIGROUPS_H_
