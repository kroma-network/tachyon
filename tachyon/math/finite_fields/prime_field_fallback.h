// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_FALLBACK_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_FALLBACK_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <string>
#include <utility>

#include "gtest/gtest_prod.h"

#include "tachyon/base/logging.h"
#include "tachyon/math/base/arithmetics.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::math {

template <typename Config>
class PrimeFieldGpu;

// A prime field is finite field GF(p) where p is a prime number.
template <typename _Config>
class PrimeField<_Config, std::enable_if_t<!_Config::kUseAsm &&
                                           (_Config::kModulusBits > 32)>>
    final : public PrimeFieldBase<PrimeField<_Config>> {
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
    PrimeField ret{};
    ret.value_ = Config::kOne;
    return ret;
  }

  static PrimeField Random() {
    return PrimeField(BigInt<N>::Random(Config::kModulus));
  }

  constexpr static std::optional<PrimeField> FromDecString(
      std::string_view str) {
    std::optional<BigInt<N>> value = BigInt<N>::FromDecString(str);
    if (!value.has_value()) return std::nullopt;
    if (value >= Config::kModulus) {
      LOG(ERROR) << "value(" << str << ") is greater than or equal to modulus";
      return std::nullopt;
    }
    return PrimeField(std::move(value).value());
  }
  constexpr static std::optional<PrimeField> FromHexString(
      std::string_view str) {
    std::optional<BigInt<N>> value = BigInt<N>::FromHexString(str);
    if (!value.has_value()) return std::nullopt;
    if (value >= Config::kModulus) {
      LOG(ERROR) << "value(" << str << ") is greater than or equal to modulus";
      return std::nullopt;
    }
    return PrimeField(std::move(value).value());
  }

  constexpr static PrimeField FromBigInt(const BigInt<N>& big_int) {
    return PrimeField(big_int);
  }

  constexpr static PrimeField FromMontgomery(const BigInt<N>& mont) {
    PrimeField ret{};
    ret.value_ = mont;
    return ret;
  }

  static PrimeField FromMpzClass(const mpz_class& value) {
    BigInt<N> big_int;
    gmp::CopyLimbs(value, big_int.limbs);
    return FromBigInt(big_int);
  }

  static void Init() { VLOG(1) << Config::kName << " initialized"; }

  const value_type& value() const { return value_; }
  size_t GetLimbSize() const { return N; }

  constexpr bool IsZero() const { return value_.IsZero(); }

  constexpr bool IsOne() const {
    for (size_t i = 0; i < N; ++i) {
      if (value_[i] != Config::kOne[i]) return false;
    }
    return true;
  }

  std::string ToString() const { return ToBigInt().ToString(); }
  std::string ToHexString(bool pad_zero = false) const {
    return ToBigInt().ToHexString(pad_zero);
  }

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

  // AdditiveSemigroup methods
  constexpr PrimeField Add(const PrimeField& other) const {
    PrimeField ret{};
    uint64_t carry = 0;
    ret.value_ = value_.Add(other.value_, carry);
    BigInt<N>::template Clamp<Config::kModulusHasSpareBit>(Config::kModulus,
                                                           &ret.value_, carry);
    return ret;
  }

  constexpr PrimeField& AddInPlace(const PrimeField& other) {
    uint64_t carry = 0;
    value_.AddInPlace(other.value_, carry);
    BigInt<N>::template Clamp<Config::kModulusHasSpareBit>(Config::kModulus,
                                                           &value_, carry);
    return *this;
  }

  constexpr PrimeField DoubleImpl() const {
    PrimeField ret{};
    uint64_t carry = 0;
    ret.value_ = value_.MulBy2(carry);
    BigInt<N>::template Clamp<Config::kModulusHasSpareBit>(Config::kModulus,
                                                           &ret.value_, carry);
    return ret;
  }

  constexpr PrimeField& DoubleImplInPlace() {
    uint64_t carry = 0;
    value_.MulBy2InPlace(carry);
    BigInt<N>::template Clamp<Config::kModulusHasSpareBit>(Config::kModulus,
                                                           &value_, carry);
    return *this;
  }

  // AdditiveGroup methods
  constexpr PrimeField Sub(const PrimeField& other) const {
    PrimeField ret{};
    if (other.value_ > value_) {
      ret.value_ = value_.Add(Config::kModulus);
      ret.value_.SubInPlace(other.value_);
    } else {
      ret.value_ = value_.Sub(other.value_);
    }
    return ret;
  }

  constexpr PrimeField& SubInPlace(const PrimeField& other) {
    if (other.value_ > value_) {
      value_.AddInPlace(Config::kModulus);
    }
    value_.SubInPlace(other.value_);
    return *this;
  }

  constexpr PrimeField Negate() const {
    PrimeField ret{};
    if (!IsZero()) {
      ret.value_ = Config::kModulus;
      ret.value_.SubInPlace(value_);
    }
    return ret;
  }

  constexpr PrimeField& NegateInPlace() {
    if (!IsZero()) {
      BigInt<N> tmp(Config::kModulus);
      tmp.SubInPlace(value_);
      value_ = tmp;
    }
    return *this;
  }

  // TODO(chokobole): Support bigendian.
  // MultiplicativeSemigroup methods
  constexpr PrimeField Mul(const PrimeField& other) const {
    PrimeField ret{};
    if constexpr (Config::kCanUseNoCarryMulOptimization) {
      DoFastMul(*this, other, ret);
    } else {
      DoSlowMul(*this, other, ret);
    }
    return ret;
  }

  constexpr PrimeField& MulInPlace(const PrimeField& other) {
    if constexpr (Config::kCanUseNoCarryMulOptimization) {
      DoFastMul(*this, other, *this);
    } else {
      DoSlowMul(*this, other, *this);
    }
    return *this;
  }

  constexpr PrimeField SquareImpl() const {
    if (N == 1) {
      return Mul(*this);
    }
    PrimeField ret{};
    DoSquareImpl(*this, ret);
    return ret;
  }

  constexpr PrimeField& SquareImplInPlace() {
    if (N == 1) {
      return MulInPlace(*this);
    }
    DoSquareImpl(*this, *this);
    return *this;
  }

  // MultiplicativeGroup methods
  constexpr std::optional<PrimeField> Inverse() const {
    PrimeField ret{};
    if (LIKELY(value_.template MontgomeryInverse<Config::kModulusHasSpareBit>(
            Config::kModulus, Config::kMontgomeryR2, ret.value_))) {
      return ret;
    }
    LOG_IF_NOT_GPU(ERROR) << "Inverse of zero attempted";
    return std::nullopt;
  }

  [[nodiscard]] constexpr std::optional<PrimeField*> InverseInPlace() {
    if (LIKELY(value_.template MontgomeryInverse<Config::kModulusHasSpareBit>(
            Config::kModulus, Config::kMontgomeryR2, value_))) {
      return this;
    }
    LOG_IF_NOT_GPU(ERROR) << "Inverse of zero attempted";
    return std::nullopt;
  }

 private:
  template <typename PrimeField>
  FRIEND_TEST(PrimeFieldCorrectnessTest, MultiplicativeOperators);

  constexpr static void DoFastMul(const PrimeField& a, const PrimeField& b,
                                  PrimeField& c) {
    BigInt<N> r;
    for (size_t i = 0; i < N; ++i) {
      MulResult<uint64_t> result;
      result = internal::u64::MulAddWithCarry(r[0], a[0], b[i]);
      r[0] = result.lo;

      uint64_t k = r[0] * Config::kInverse64;
      MulResult<uint64_t> result2;
      result2 = internal::u64::MulAddWithCarry(r[0], k, Config::kModulus[0]);

      for (size_t j = 1; j < N; ++j) {
        result = internal::u64::MulAddWithCarry(r[j], a[j], b[i], result.hi);
        r[j] = result.lo;
        result2 = internal::u64::MulAddWithCarry(r[j], k, Config::kModulus[j],
                                                 result2.hi);
        r[j - 1] = result2.lo;
      }
      r[N - 1] = result.hi + result2.hi;
    }
    c.value_ = r;
    BigInt<N>::template Clamp<Config::kModulusHasSpareBit>(Config::kModulus,
                                                           &c.value_, 0);
  }

  constexpr static void DoSlowMul(const PrimeField& a, const PrimeField& b,
                                  PrimeField& c) {
    BigInt<2 * N> r = a.value_.MulExtend(b.value_);
    BigInt<N>::template MontgomeryReduce64<Config::kModulusHasSpareBit>(
        r, Config::kModulus, Config::kInverse64, &c.value_);
  }

  constexpr static void DoSquareImpl(const PrimeField& a, PrimeField& b) {
    BigInt<2 * N> r;
    MulResult<uint64_t> mul_result;
    for (size_t i = 0; i < N - 1; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        mul_result =
            internal::u64::MulAddWithCarry(r[i + j], a[i], a[j], mul_result.hi);
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
      mul_result =
          internal::u64::MulAddWithCarry(r[2 * i], a[i], a[i], mul_result.hi);
      r[2 * i] = mul_result.lo;
      add_result = internal::u64::AddWithCarry(r[2 * i + 1], mul_result.hi);
      r[2 * i + 1] = add_result.result;
      mul_result.hi = add_result.carry;
    }
    BigInt<N>::template MontgomeryReduce64<Config::kModulusHasSpareBit>(
        r, Config::kModulus, Config::kInverse64, &b.value_);
  }

  BigInt<N> value_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_FALLBACK_H_
