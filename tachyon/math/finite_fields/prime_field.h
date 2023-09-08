#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_

#include <stddef.h>
#include <stdint.h>

#include <ostream>
#include <string>

#include "gtest/gtest_prod.h"

#include "tachyon/math/base/arithmetics.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/finite_fields/modulus.h"
#include "tachyon/math/finite_fields/prime_field_base.h"
#include "tachyon/math/finite_fields/prime_field_forward.h"

namespace tachyon::math {

template <typename Config>
class PrimeFieldGpu;

template <typename _Config>
class PrimeField<_Config, std::enable_if_t<!_Config::kIsSpecialPrime>>
    : public PrimeFieldBase<PrimeField<_Config>> {
 public:
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;

  using Config = _Config;
  using BigIntTy = BigInt<N>;
  using value_type = BigInt<N>;

  using CpuField = PrimeField<Config>;
  using GpuField = PrimeFieldGpu<Config>;

  constexpr PrimeField() = default;
  template <typename T,
            std::enable_if_t<std::is_constructible_v<BigInt<N>, T>>* = nullptr>
  constexpr explicit PrimeField(T value) : PrimeField(BigInt<N>(value)) {}
  constexpr explicit PrimeField(const BigInt<N>& value) : value_(value) {
    DCHECK_LT(value_, Config::kModulus);
    PrimeField p;
    p.value_ = Config::kMontgomeryR2;
    MulInPlace(p);
  }
  constexpr PrimeField(const PrimeField& other) = default;
  constexpr PrimeField& operator=(const PrimeField& other) = default;
  constexpr PrimeField(PrimeField&& other) = default;
  constexpr PrimeField& operator=(PrimeField&& other) = default;

  constexpr static PrimeField Zero() { return PrimeField(); }

  constexpr static PrimeField One() {
    PrimeField ret;
    ret.value_ = Config::kOne;
    return ret;
  }

  static PrimeField Random() {
    return PrimeField(BigInt<N>::Random(Config::kModulus));
  }

  constexpr static PrimeField FromDecString(std::string_view str) {
    return PrimeField(BigInt<N>::FromDecString(str));
  }
  constexpr static PrimeField FromHexString(std::string_view str) {
    return PrimeField(BigInt<N>::FromHexString(str));
  }

  constexpr static PrimeField FromBigInt(const BigInt<N>& big_int) {
    return PrimeField(big_int);
  }

  static PrimeField FromMontgomery(const BigInt<N>& big_int) {
    PrimeField ret;
    ret.value_ = big_int;
    return ret;
  }

  static PrimeField FromMpzClass(const mpz_class& value) {
    BigInt<N> big_int;
    gmp::CopyLimbs(value, big_int.limbs);
    return FromBigInt(big_int);
  }

  static void Init() {
    // Do nothing.
  }

  const value_type& value() const { return value_; }
  size_t GetLimbSize() const { return N; }

  constexpr bool IsZero() const { return value_.IsZero(); }

  constexpr bool IsOne() const { return ToBigInt().IsOne(); }

  std::string ToString() const { return ToBigInt().ToString(); }
  std::string ToHexString() const { return ToBigInt().ToHexString(); }

  mpz_class ToMpzClass() const {
    mpz_class ret;
    gmp::WriteLimbs(ToBigInt().limbs, N, &ret);
    return ret;
  }

  // TODO(chokobole): Support bigendian.
  constexpr BigInt<N> ToBigInt() const {
    return BigInt<N>::FromMontgomery64(value_, Config::kModulus,
                                       Config::kInverse64);
  }

  constexpr const BigInt<N>& ToMontgomery() const { return value_; }

  constexpr uint64_t& operator[](size_t i) { return value_[i]; }
  constexpr const uint64_t& operator[](size_t i) const { return value_[i]; }

  constexpr bool operator==(const PrimeField& other) const {
    return ToBigInt() == other.ToBigInt();
  }

  constexpr bool operator!=(const PrimeField& other) const {
    return ToBigInt() != other.ToBigInt();
  }

  constexpr bool operator<(const PrimeField& other) const {
    return ToBigInt() < other.ToBigInt();
  }

  constexpr bool operator>(const PrimeField& other) const {
    return ToBigInt() > other.ToBigInt();
  }

  constexpr bool operator<=(const PrimeField& other) const {
    return ToBigInt() <= other.ToBigInt();
  }

  constexpr bool operator>=(const PrimeField& other) const {
    return ToBigInt() >= other.ToBigInt();
  }

  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  BigInt<N> DivBy2Exp(uint32_t exp) const {
    return ToBigInt().DivBy2ExpInPlace(exp);
  }

  // AdditiveSemigroup methods
  constexpr PrimeField& AddInPlace(const PrimeField& other) {
    uint64_t carry = 0;
    value_.AddInPlace(other.value_, carry);
    BigInt<N>::template Clamp<Config::kModulusHasSpareBit>(Config::kModulus,
                                                           &value_, carry);
    return *this;
  }

  constexpr PrimeField& DoubleInPlace() {
    uint64_t carry = 0;
    value_.MulBy2InPlace(carry);
    BigInt<N>::template Clamp<Config::kModulusHasSpareBit>(Config::kModulus,
                                                           &value_, carry);
    return *this;
  }

  // AdditiveGroup methods
  constexpr PrimeField& SubInPlace(const PrimeField& other) {
    if (other.value_ > value_) {
      value_.AddInPlace(Config::kModulus);
    }
    value_.SubInPlace(other.value_);
    return *this;
  }

  constexpr PrimeField& NegInPlace() {
    if (!IsZero()) {
      BigInt<N> tmp(Config::kModulus);
      tmp.SubInPlace(value_);
      value_ = tmp;
    }
    return *this;
  }

  // TODO(chokobole): Support bigendian.
  // MultiplicativeSemigroup methods
  constexpr PrimeField& MulInPlace(const PrimeField& other) {
    if constexpr (Config::kCanUseNoCarryMulOptimization) {
      return FastMulInPlace(other);
    } else {
      return SlowMulInPlace(other);
    }
  }

  constexpr PrimeField& SquareInPlace() {
    if (N == 1) {
      return MulInPlace(*this);
    }

    BigInt<N * 2> r;
    MulResult<uint64_t> mul_result;
    for (size_t i = 0; i < N - 1; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        mul_result = internal::u64::MulAddWithCarry(r[i + j], value_[i],
                                                    value_[j], mul_result.hi);
        r[i + j] = mul_result.lo;
      }
      r[i + N] = mul_result.hi;
      mul_result.hi = 0;
    }

    r[2 * N - 1] = r[2 * N - 2] >> 63;
    for (size_t i = 2; i < 2 * N - 1; ++i) {
      r[2 * N - i] = (r[2 * N - i] << 1) | (r[2 * N - (i + 1)] >> 63);
    }
    r[1] <<= 1;

    AddResult<uint64_t> add_result;
    for (size_t i = 0; i < N; ++i) {
      mul_result = internal::u64::MulAddWithCarry(r[2 * i], value_[i],
                                                  value_[i], mul_result.hi);
      r[2 * i] = mul_result.lo;
      add_result = internal::u64::AddWithCarry(r[2 * i + 1], mul_result.hi);
      r[2 * i + 1] = add_result.result;
      mul_result.hi = add_result.carry;
    }
    BigInt<N>::template MontgomeryReduce64<Config::kModulusHasSpareBit>(
        r, Config::kModulus, Config::kInverse64, &value_);
    return *this;
  }

  // MultiplicativeGroup methods
  constexpr PrimeField& InverseInPlace() {
    value_ = value_.template MontgomeryInverse<Config::kModulusHasSpareBit>(
        Config::kModulus, Config::kMontgomeryR2);
    return *this;
  }

 private:
  template <typename PrimeFieldType>
  FRIEND_TEST(PrimeFieldCorrectnessTest, MultiplicativeOperators);

  constexpr PrimeField& FastMulInPlace(const PrimeField& other) {
    BigInt<N> r;
    for (size_t i = 0; i < N; ++i) {
      MulResult<uint64_t> result;
      result = internal::u64::MulAddWithCarry(r[0], value_[0], other.value_[i]);
      r[0] = result.lo;

      uint64_t k = r[0] * Config::kInverse64;
      MulResult<uint64_t> result2;
      result2 = internal::u64::MulAddWithCarry(r[0], k, Config::kModulus[0]);

      for (size_t j = 1; j < N; ++j) {
        result = internal::u64::MulAddWithCarry(r[j], value_[j],
                                                other.value_[i], result.hi);
        r[j] = result.lo;
        result2 = internal::u64::MulAddWithCarry(r[j], k, Config::kModulus[j],
                                                 result2.hi);
        r[j - 1] = result2.lo;
      }
      r[N - 1] = result.hi + result2.hi;
    }
    value_ = r;
    BigInt<N>::template Clamp<Config::kModulusHasSpareBit>(Config::kModulus,
                                                           &value_, 0);
    return *this;
  }

  constexpr PrimeField& SlowMulInPlace(const PrimeField& other) {
    BigInt<N * 2> r;
    MulResult<uint64_t> mul_result;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        mul_result = internal::u64::MulAddWithCarry(
            r[i + j], value_[i], other.value_[j], mul_result.hi);
        r[i + j] = mul_result.lo;
      }
      r[i + N] = mul_result.hi;
      mul_result.hi = 0;
    }
    BigInt<N>::template MontgomeryReduce64<Config::kModulusHasSpareBit>(
        r, Config::kModulus, Config::kInverse64, &value_);
    return *this;
  }

  BigInt<N> value_;
};

template <typename Config>
std::ostream& operator<<(std::ostream& os, const PrimeField<Config>& f) {
  return os << f.ToString();
}

template <typename Config>
class MultiplicativeIdentity<PrimeField<Config>> {
 public:
  using F = PrimeField<Config>;

  static const F& One() {
    static F one(F::One());
    return one;
  }

  constexpr static bool IsOne(const F& value) { return value.IsOne(); }
};

template <typename Config>
class AdditiveIdentity<PrimeField<Config>> {
 public:
  using F = PrimeField<Config>;

  static const F& Zero() {
    static F zero(F::Zero());
    return zero;
  }

  constexpr static bool IsZero(const F& value) { return value.IsZero(); }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_H_
