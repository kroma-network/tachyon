#ifndef TACHYON_MATH_BASE_SEMIGROUPS_H_
#define TACHYON_MATH_BASE_SEMIGROUPS_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/bits.h"
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

#define SUPPORTS_UNARY_OPERATOR(Name)                                        \
  template <typename T, typename = void>                                     \
  struct Supports##Name : std::false_type {};                                \
                                                                             \
  template <typename T>                                                      \
  struct Supports##Name<T, decltype(void(std::declval<T>().Name()))>         \
      : std::true_type {};                                                   \
                                                                             \
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
SUPPORTS_UNARY_OPERATOR(SquareImpl);
SUPPORTS_BINARY_OPERATOR(Add);
SUPPORTS_UNARY_OPERATOR(DoubleImpl);

template <typename T, typename = void>
struct SupportsSize : std::false_type {};

template <typename T>
struct SupportsSize<T, decltype(void(std::size(std::declval<T>())))>
    : std::true_type {};

template <typename G>
struct MultiplicativeSemigroupTraits {
  using ReturnTy = G;
};

template <typename G, typename SFINAE = void>
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
  using MulResult =
      typename internal::MultiplicativeSemigroupTraits<G>::ReturnTy;

  // Multiplication: a * b
  template <typename G2,
            std::enable_if_t<internal::SupportsMul<G, G2>::value>* = nullptr>
  constexpr auto operator*(const G2& other) const {
    const G* g = static_cast<const G*>(this);
    return g->Mul(other);
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
    if constexpr (internal::SupportsSquareImpl<G>::value) {
      const G* g = static_cast<const G*>(this);
      return g->SquareImpl();
    } else {
      return operator*(static_cast<const G&>(*this));
    }
  }

  // a.SquareInPlace(): a = a²
  constexpr G& SquareInPlace() {
    if constexpr (internal::SupportsSquareImplInPlace<G>::value) {
      G* g = static_cast<G*>(this);
      return g->SquareImplInPlace();
    } else if constexpr (internal::SupportsMulInPlace<G, G>::value) {
      return operator*=(static_cast<const G&>(*this));
    } else {
      static_assert(base::AlwaysFalse<G>);
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
    } else {
      return DoPow(scalar.ToBigInt());
    }
  }

  template <uint32_t Power>
  [[nodiscard]] constexpr auto ConstPow() const {
    const G& g = static_cast<const G&>(*this);
    if constexpr (Power == 0)
      return MulResult::One();
    else if constexpr (Power == 1)
      return g;
    else if constexpr (Power == 2)
      return Square();
    else if constexpr (Power == 3)
      return Square() * g;
    else if constexpr (Power == 4)
      return Square().Square();
    else if constexpr (Power == 5) {
      MulResult g4 = Square();
      g4.SquareInPlace();
      return g4 * g;
    } else if constexpr (Power == 6) {
      MulResult g2 = Square();
      MulResult g4 = g2;
      g4.SquareInPlace();
      return g4 * g2;
    } else if constexpr (Power == 7) {
      MulResult g2 = Square();
      MulResult g4 = g2;
      g4.SquareInPlace();
      return g4 * g2 * g;
    } else {
      return DoPow(BigInt<1>(Power));
    }
  }

  // Computes the power of a base element using a pre-computed table of powers
  // of two, instead of performing repeated multiplications.
  template <size_t N>
  static MulResult PowWithTable(absl::Span<const G> powers_of_2,
                                const BigInt<N>& exponent) {
    auto it = BitIteratorLE<BigInt<N>>::begin(&exponent);
    auto end = BitIteratorLE<BigInt<N>>::end(&exponent, true);
    MulResult g = MulResult::One();
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
  constexpr static std::vector<MulResult> GetSuccessivePowers(
      size_t size, const G& generator, const G& c = G::One()) {
    std::vector<MulResult> ret(size);
    base::Parallelize(
        ret, [&generator, &c](absl::Span<G> chunk, size_t chunk_offset,
                              size_t chunk_size) {
          MulResult pow = generator.Pow(chunk_offset * chunk_size);
          if (!c.IsOne()) pow *= c;

          // NOTE: It is not possible to have empty chunk so this is safe
          for (size_t i = 0; i < chunk.size() - 1; ++i) {
            chunk[i] = pow;
            pow *= generator;
          }
          chunk.back() = std::move(pow);
        });
    return ret;
  }

  // Refer to |GetSuccessivePowers()| for basic approach.
  // Populates bit-reversed index instead of given index in a parallelized way.
  constexpr static std::vector<MulResult> GetBitRevIndexSuccessivePowers(
      size_t size, const G& generator, const G& c = G::One()) {
    std::vector<MulResult> ret(size);
    uint32_t log_size = base::bits::CheckedLog2(size);
    base::Parallelize(
        ret, [log_size, &generator, &c, &ret](
                 absl::Span<G> chunk, size_t chunk_offset, size_t chunk_size) {
          size_t chunk_start = chunk_offset * chunk_size;
          MulResult pow = generator.Pow(chunk_start);
          if (!c.IsOne()) pow *= c;
          for (size_t idx = chunk_start; idx < chunk_start + chunk.size() - 1;
               ++idx) {
            size_t ridx = base::bits::ReverseBitsLen(idx, log_size);
            ret[ridx] = pow;
            pow *= generator;
          }
          ret[base::bits::ReverseBitsLen(chunk_start + chunk.size() - 1,
                                         log_size)] = std::move(pow);
        });
    return ret;
  }

  // Refer to |GetSuccessivePowers()| for basic approach.
  // Populates bit-reversed index instead of given index in a serial way.
  constexpr static std::vector<MulResult> GetBitRevIndexSuccessivePowersSerial(
      size_t size, const G& generator, const G& c = G::One()) {
    std::vector<MulResult> ret(size);
    uint32_t log_size = base::bits::CheckedLog2(size);
    MulResult pow = c.IsOne() ? G::One() : c;
    for (size_t idx = 0; idx < size - 1; ++idx) {
      ret[base::bits::ReverseBitsLen(idx, log_size)] = pow;
      pow *= generator;
    }
    ret[base::bits::ReverseBitsLen(size - 1, log_size)] = std::move(pow);
    return ret;
  }

  constexpr auto ExpPowOfTwo(uint32_t log_n) const {
    G val = *static_cast<const G*>(this);
    for (size_t i = 0; i < log_n; ++i) {
      val.SquareInPlace();
    }
    return val;
  }

 private:
  constexpr static size_t kDefaultParallelThreshold = 1024;

  template <size_t N>
  [[nodiscard]] constexpr MulResult DoPow(const BigInt<N>& exponent) const {
    const G* g = static_cast<const G*>(this);
    MulResult ret = MulResult::One();
    auto it = BitIteratorBE<BigInt<N>>::begin(&exponent, true);
    auto end = BitIteratorBE<BigInt<N>>::end(&exponent);
    while (it != end) {
      if constexpr (internal::SupportsSquareImplInPlace<G>::value ||
                    internal::SupportsMulInPlace<G, G>::value) {
        ret.SquareInPlace();
      } else {
        ret = ret.Square();
      }
      if (*it) {
        if constexpr (internal::SupportsMulInPlace<MulResult, G>::value) {
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
  using AddResult = typename internal::AdditiveSemigroupTraits<G>::ReturnTy;

  // Addition: a + b
  template <typename G2,
            std::enable_if_t<internal::SupportsAdd<G, G2>::value>* = nullptr>
  constexpr auto operator+(const G2& other) const {
    const G* g = static_cast<const G*>(this);
    return g->Add(other);
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
    if constexpr (internal::SupportsDoubleImpl<G>::value) {
      const G* g = static_cast<const G*>(this);
      return g->DoubleImpl();
    } else {
      return operator+(static_cast<const G&>(*this));
    }
  }

  // a.DoubleInPlace(): a = 2a
  constexpr G& DoubleInPlace() {
    if constexpr (internal::SupportsDoubleImplInPlace<G>::value) {
      G* g = static_cast<G*>(this);
      return g->DoubleImplInPlace();
    } else if constexpr (internal::SupportsAddInPlace<G, G>::value) {
      return operator+=(static_cast<const G&>(*this));
    } else {
      static_assert(base::AlwaysFalse<G>);
    }
  }

  // FIXME(chokobole): It would be nice to support multiplication operator
  // when multiplying scalar. But in g++
  // (Ubuntu 11.3.0-1ubuntu1~22.04.1) 11.3.0, the commented line isn't compiled
  // with this error.
  // "error: request for member 'operator*' is ambiguous".
  // constexpr auto operator*(const BigInt<N>& scalar) const {
  template <size_t N>
  [[nodiscard]] constexpr auto ScalarMul(const BigInt<N>& scalar) const {
    return DoScalarMul(scalar);
  }

  template <typename Scalar>
  [[nodiscard]] constexpr auto ScalarMul(const Scalar& scalar) const {
    if constexpr (std::is_constructible_v<BigInt<1>, Scalar>) {
      return DoScalarMul(BigInt<1>(scalar));
    } else {
      return DoScalarMul(scalar.ToBigInt());
    }
  }

  template <uint64_t Scalar>
  [[nodiscard]] constexpr auto ConstScalarMul() const {
    const G& g = static_cast<const G&>(*this);
    if constexpr (Scalar == 0)
      return AddResult::Zero();
    else if constexpr (Scalar == 1)
      return g;
    else if constexpr (Scalar == 2)
      return Double();
    else if constexpr (Scalar == 3)
      return Double() + g;
    else if constexpr (Scalar == 4)
      return Double().Double();
    else if constexpr (Scalar == 5) {
      AddResult g4 = Double();
      g4.DoubleInPlace();
      return g4 + g;
    } else if constexpr (Scalar == 6) {
      AddResult g2 = Double();
      AddResult g4 = g2;
      g4.DoubleInPlace();
      return g4 + g2;
    } else if constexpr (Scalar == 7) {
      AddResult g2 = Double();
      AddResult g4 = g2;
      g4.DoubleInPlace();
      return g4 + g2 + g;
    } else {
      return DoScalarMul(BigInt<1>(Scalar));
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
    } else {
      return MultiScalarMulSSMB(scalar_or_scalars, base_or_bases, outputs);
    }
  }

  // generator: G
  // return: [0G, 1G, 2G, ..., (|size| - 1)G]
  // NOTE(chokobole): Unlike |GetSuccessivePowers()|, this doesn't have an
  // additional |c| parameter because there's no usecase that depends on
  // it.
  constexpr static std::vector<AddResult> GetSuccessiveScalarMuls(
      size_t size, const G& generator) {
    std::vector<AddResult> ret(size);
    base::Parallelize(
        ret, [&generator](absl::Span<G> chunk, size_t chunk_offset,
                          size_t chunk_size) {
          AddResult scalar_mul = generator.ScalarMul(chunk_offset * chunk_size);

          // NOTE: It is not possible to have empty chunk so this is safe
          for (size_t i = 0; i < chunk.size() - 1; ++i) {
            chunk[i] = scalar_mul;
            scalar_mul += generator;
          }
          chunk.back() = std::move(scalar_mul);
        });
    return ret;
  }

  // Linear combination
  // - forward: a₀ * rⁿ⁻¹ + a₁ * rⁿ⁻² + ... + aₙ₋₁
  // - backward: a₀ + a₁ * r + ... + aₙ₋₁ * rⁿ⁻¹
  // NOTE(chokobole): For performance reasons, we recommend using
  // |LinearCombinationInPlace()| if possible.
  template <bool Forward, typename Container, typename T>
  constexpr static AddResult LinearCombination(const Container& values,
                                               const T& r) {
    size_t size = std::size(values);
    AddResult ret = AddResult::Zero();
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
            typename AddResult =
                typename internal::AdditiveSemigroupTraits<G>::ReturnTy,
            std::enable_if_t<std::is_same_v<G, AddResult>>* = nullptr>
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
  [[nodiscard]] constexpr AddResult DoScalarMul(const BigInt<N>& scalar) const {
    const G* g = static_cast<const G*>(this);
    AddResult ret = AddResult::Zero();
    auto it = BitIteratorBE<BigInt<N>>::begin(&scalar, true);
    auto end = BitIteratorBE<BigInt<N>>::end(&scalar);
    while (it != end) {
      if constexpr (internal::SupportsDoubleImplInPlace<G>::value ||
                    internal::SupportsAddInPlace<G, G>::value) {
        ret.DoubleInPlace();
      } else {
        ret = ret.Double();
      }
      if (*it) {
        if constexpr (internal::SupportsAddInPlace<AddResult, G>::value) {
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
  CONSTEXPR_IF_NOT_OPENMP static bool MultiScalarMulMSMB(
      const ScalarContainer& scalars, const BaseContainer& bases,
      OutputContainer* outputs) {
    size_t size = scalars.size();
    if (size != std::size(bases)) return false;
    if (size != std::size(*outputs)) return false;
    if (size == 0) {
      LOG(ERROR) << "scalars and bases are empty";
      return false;
    }
    OMP_PARALLEL_FOR(size_t i = 0; i < size; ++i) {
      (*outputs)[i] = bases[i].ScalarMul(scalars[i]);
    }
    return true;
  }

  // Multi Scalar Single Base
  template <typename ScalarContainer, typename OutputContainer>
  CONSTEXPR_IF_NOT_OPENMP static bool MultiScalarMulMSSB(
      const ScalarContainer& scalars, const G& base, OutputContainer* outputs) {
    size_t size = std::size(scalars);
    if (size != std::size(*outputs)) return false;
    if (size == 0) {
      LOG(ERROR) << "scalars are empty";
      return false;
    }
    OMP_PARALLEL_FOR(size_t i = 0; i < size; ++i) {
      (*outputs)[i] = base.ScalarMul(scalars[i]);
    }
    return true;
  }

  // Single Scalar Multi Base
  template <typename Scalar, typename BaseContainer, typename OutputContainer>
  CONSTEXPR_IF_NOT_OPENMP static bool MultiScalarMulSSMB(
      const Scalar& scalar, const BaseContainer& bases,
      OutputContainer* outputs) {
    size_t size = std::size(bases);
    if (size != std::size(*outputs)) return false;
    if (size == 0) {
      LOG(ERROR) << "bases are empty";
      return false;
    }
    OMP_PARALLEL_FOR(size_t i = 0; i < size; ++i) {
      (*outputs)[i] = bases[i].ScalarMul(scalar);
    }
    return true;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_SEMIGROUPS_H_
