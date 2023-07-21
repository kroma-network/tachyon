#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_CUDA_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_CUDA_H_

#include <stddef.h>
#include <stdint.h>

#include <ostream>
#include <string>

#include "tachyon/base/random.h"
#include "tachyon/math/base/arithmetics.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/finite_fields/kernels/carry_chain.cu.h"
#include "tachyon/math/finite_fields/modulus.h"
#include "tachyon/math/finite_fields/prime_field_base.h"
#include "tachyon/math/finite_fields/prime_field_mont.h"

namespace tachyon {
namespace math {

template <typename _Config>
class PrimeFieldMontCuda : public PrimeFieldBase<PrimeFieldMontCuda<_Config>> {
 public:
  static constexpr size_t kModulusBits = _Config::kModulusBits;
  static constexpr size_t kLimbNums = (kModulusBits + 63) / 64;
  static constexpr size_t N = kLimbNums;

  using Config = _Config;
  using value_type = BigInt<N>;

  static constexpr bool kModulusHasSparseBit =
      Modulus<N>::HasSparseBit(Config::kModulus);
  static constexpr bool kCanUseNoCarryMulOptimization =
      Modulus<N>::CanUseNoCarryMulOptimization(Config::kModulus);
  static constexpr BigInt<N> kMontgomeryR =
      Modulus<N>::MontgomeryR(Config::kModulus);
  static constexpr BigInt<N> kMontgomeryR2 =
      Modulus<N>::MontgomeryR2(Config::kModulus);
  static constexpr uint64_t kInverse64 =
      Modulus<N>::template Inverse<uint64_t>(Config::kModulus);
  static constexpr uint32_t kInverse32 =
      Modulus<N>::template Inverse<uint32_t>(Config::kModulus);

  constexpr PrimeFieldMontCuda() = default;
  template <typename T,
            std::enable_if_t<std::is_constructible_v<BigInt<N>, T>>* = nullptr>
  constexpr explicit PrimeFieldMontCuda(T value)
      : PrimeFieldMontCuda(BigInt<N>(value)) {}
  constexpr explicit PrimeFieldMontCuda(const BigInt<N>& value) {
    DCHECK_LT(value, GetModulus());
    PrimeFieldMont<Config> p(value);
    value_ = p.value();
  }
  constexpr PrimeFieldMontCuda(const PrimeFieldMontCuda& other) = default;
  constexpr PrimeFieldMontCuda& operator=(const PrimeFieldMontCuda& other) =
      default;
  constexpr PrimeFieldMontCuda(PrimeFieldMontCuda&& other) = default;
  constexpr PrimeFieldMontCuda& operator=(PrimeFieldMontCuda&& other) = default;

  __host__ __device__ constexpr static PrimeFieldMontCuda Zero() {
    return PrimeFieldMontCuda(BigInt<N>::Zero());
  }

  __host__ __device__ constexpr static PrimeFieldMontCuda One() {
    return PrimeFieldMontCuda(BigInt<N>::One());
  }

  static PrimeFieldMontCuda Random() {
    BigInt<N> big_int;
    for (size_t i = 0; i < N; ++i) {
      big_int.limbs[i] = base::Uniform<uint64_t, uint64_t>(
          0, std::numeric_limits<uint64_t>::max());
    }
    while (big_int >= GetModulus()) {
      big_int.DivBy2InPlace();
    }
    return PrimeFieldMontCuda(big_int);
  }

  constexpr static PrimeFieldMontCuda FromDecString(std::string_view str) {
    return PrimeFieldMontCuda(BigInt<N>::FromDecString(str));
  }
  constexpr static PrimeFieldMontCuda FromHexString(std::string_view str) {
    return PrimeFieldMontCuda(BigInt<N>::FromHexString(str));
  }

  constexpr static PrimeFieldMontCuda FromBigInt(const BigInt<N>& big_int) {
    return PrimeFieldMontCuda(big_int);
  }

  constexpr static PrimeFieldMontCuda FromMontgomery(const BigInt<N>& big_int) {
    PrimeFieldMontCuda ret;
    ret.value_ = big_int;
    return ret;
  }

  __host__ __device__ static constexpr BigInt<N> GetModulus() {
    return Config::kModulus;
  }

  __host__ __device__ const value_type& value() const { return value_; }

  __host__ __device__ constexpr bool IsZero() const {
    const uint64_t* x = value_.limbs;
    uint64_t limbs_or = x[0];
    for (size_t i = 1; i < N; ++i) limbs_or |= x[i];
    return limbs_or == 0;
  }

  __host__ __device__ constexpr bool IsOne() const {
    BigInt<N> value = ToBigInt();
    const uint64_t* x = value.limbs;
    uint64_t limbs_or = 0;
    for (size_t i = 1; i < N; ++i) limbs_or |= x[i];
    return x[0] == 1 && limbs_or == 0;
  }

  __host__ __device__ constexpr bool IsEven() const { return value_.IsEven(); }

  __host__ __device__ constexpr bool IsOdd() const { return value_.IsOdd(); }

  std::string ToString() const { return ToBigInt().ToString(); }
  std::string ToHexString() const { return ToBigInt().ToHexString(); }

  mpz_class ToMpzClass() const {
    mpz_class ret;
    gmp::WriteLimbs(ToBigInt().limbs, N, &ret);
    return ret;
  }

  constexpr BigInt<N> ToBigInt() const {
    return BigInt<N>::FromMontgomery(value_, Config::kModulus, kInverse64);
  }

  constexpr const BigInt<N>& ToMontgomery() const { return value_; }

  __host__ __device__ constexpr bool operator==(
      const PrimeFieldMontCuda& other) const {
    const uint64_t* x = value_.limbs;
    const uint64_t* y = other.value_.limbs;
    uint64_t limbs_or = x[0] ^ y[0];
    for (size_t i = 1; i < N; ++i) limbs_or |= x[i] ^ y[i];
    return limbs_or == 0;
  }

  __host__ __device__ constexpr bool operator!=(
      const PrimeFieldMontCuda& other) const {
    return !operator==(other);
  }

  __device__ constexpr bool operator<(const PrimeFieldMontCuda& other) const {
    PrimeFieldMontCuda results;
    uint64_t carry = SubLimbs<true>(value_, other.value_, results.value_);
    return carry;
  }

  __device__ constexpr bool operator>(const PrimeFieldMontCuda& other) const {
    PrimeFieldMontCuda results;
    uint64_t carry = SubLimbs<true>(other.value_, value_, results.value_);
    return carry;
  }

  __device__ constexpr bool operator<=(const PrimeFieldMontCuda& other) const {
    return !operator>(other);
  }

  __device__ constexpr bool operator>=(const PrimeFieldMontCuda& other) const {
    return !operator<(other);
  }

  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  mpz_class DivBy2Exp(uint64_t exp) const {
    return gmp::DivBy2Exp(ToMpzClass(), exp);
  }

  // AdditiveSemigroup methods
  __device__ constexpr PrimeFieldMontCuda& AddInPlace(
      const PrimeFieldMontCuda& other) {
    AddLimbs<false>(value_, other.value_, value_);
    *this = Clamp(*this);
    return *this;
  }

  __device__ constexpr PrimeFieldMontCuda& DoubleInPlace() {
    return AddInPlace(*this);
  }

  // AdditiveGroup methods
  __device__ constexpr PrimeFieldMontCuda& SubInPlace(
      const PrimeFieldMontCuda& other) {
    uint64_t carry = SubLimbs<true>(value_, other.value_, value_);
    if (carry == 0) return *this;
    AddLimbs<false>(value_, GetModulus(), value_);
    return *this;
  }

  __device__ constexpr PrimeFieldMontCuda& NegInPlace() {
    NOTIMPLEMENTED();
    return *this;
  }

  // MultiplicativeSemigroup methods
  // NOTE(chokobole): This needs __host__ to allow to construct
  // PrimeFieldMontCuda from host side.
  __host__ __device__ constexpr PrimeFieldMontCuda& MulInPlace(
      const PrimeFieldMontCuda& other) {
    // NOTIMPLEMENTED();
    return *this;
  }

  __device__ constexpr PrimeFieldMontCuda& SquareInPlace() {
    return MulInPlace(*this);
  }

  // MultiplicativeGroup methods
  __device__ constexpr PrimeFieldMontCuda& InverseInPlace() {
    NOTIMPLEMENTED();
    return *this;
  }

 private:
  template <bool CarryOut>
  __device__ constexpr static uint64_t AddLimbs(const BigInt<N>& xs,
                                                const BigInt<N>& ys,
                                                BigInt<N>& results) {
    const uint64_t* x = xs.limbs;
    const uint64_t* y = ys.limbs;
    uint64_t* r = results.limbs;
    CarryChain<CarryOut ? N + 1 : N> chain;
    for (size_t i = 0; i < N; ++i) {
      r[i] = chain.Add(x[i], y[i]);
    }
    if constexpr (!CarryOut) return 0;
    return chain.Add(0, 0);
  }

  template <bool CarryOut>
  __device__ constexpr static uint64_t SubLimbs(const BigInt<N>& xs,
                                                const BigInt<N>& ys,
                                                BigInt<N>& results) {
    const uint64_t* x = xs.limbs;
    const uint64_t* y = ys.limbs;
    uint64_t* r = results.limbs;
    CarryChain<CarryOut ? N + 1 : N> chain;
    for (size_t i = 0; i < N; ++i) {
      r[i] = chain.Sub(x[i], y[i]);
    }
    if constexpr (!CarryOut) return 0;
    return chain.Sub(0, 0);
  }

  __device__ constexpr static PrimeFieldMontCuda Clamp(PrimeFieldMontCuda& xs) {
    PrimeFieldMontCuda results;
    return SubLimbs<true>(xs.value_, GetModulus(), results.value_) ? xs
                                                                   : results;
  }

  BigInt<N> value_;
};

template <typename Config>
std::ostream& operator<<(std::ostream& os,
                         const PrimeFieldMontCuda<Config>& f) {
  return os << f.ToString();
}

template <typename Config>
class MultiplicativeIdentity<PrimeFieldMontCuda<Config>> {
 public:
  using F = PrimeFieldMontCuda<Config>;

  static const F& One() {
    static F one(F::One());
    return one;
  }

  constexpr static bool IsOne(const F& value) { return value.IsOne(); }
};

template <typename Config>
class AdditiveIdentity<PrimeFieldMontCuda<Config>> {
 public:
  using F = PrimeFieldMontCuda<Config>;

  static const F& Zero() {
    static F zero(F::Zero());
    return zero;
  }

  constexpr static bool IsZero(const F& value) { return value.IsZero(); }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_CUDA_H_
