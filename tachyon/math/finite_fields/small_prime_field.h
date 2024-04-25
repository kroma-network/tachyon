// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_SMALL_PRIME_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_SMALL_PRIME_FIELD_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <string>

#include "tachyon/base/logging.h"
#include "tachyon/base/random.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/base/egcd.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::math {

template <typename Config>
class PrimeFieldGpu;

// A prime field is finite field GF(p) where p is a prime number.
template <typename _Config>
class PrimeField<_Config, std::enable_if_t<!_Config::kIsSpecialPrime &&
                                           (_Config::kModulusBits <= 32)>>
    final : public PrimeFieldBase<PrimeField<_Config>> {
 public:
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;

  using Config = _Config;
  using BigIntTy = BigInt<N>;
  using MontgomeryTy = BigInt<N>;
  using value_type = uint32_t;

  using CpuField = PrimeField<Config>;
  using GpuField = PrimeFieldGpu<Config>;

  constexpr PrimeField() = default;
  constexpr explicit PrimeField(uint32_t value) : value_(value) {
    DCHECK_LT(value_, GetModulus());
  }
  constexpr explicit PrimeField(const BigInt<N>& value) : PrimeField(value[0]) {
    DCHECK_LT(value_, GetModulus());
  }
  constexpr PrimeField(const PrimeField& other) = default;
  constexpr PrimeField& operator=(const PrimeField& other) = default;
  constexpr PrimeField(PrimeField&& other) = default;
  constexpr PrimeField& operator=(PrimeField&& other) = default;

  constexpr static PrimeField Zero() { return PrimeField(); }
  constexpr static PrimeField One() { return PrimeField(1); }

  static PrimeField Random() {
    return PrimeField(
        base::Uniform(base::Range<uint32_t>::Until(GetModulus())));
  }

  static std::optional<PrimeField> FromDecString(std::string_view str) {
    uint64_t value;
    if (!base::StringToUint64(str, &value)) return std::nullopt;
    if (value >= uint64_t{GetModulus()}) {
      LOG(ERROR) << "value(" << str << ") is greater than or equal to modulus";
      return std::nullopt;
    }
    return PrimeField(value);
  }
  static std::optional<PrimeField> FromHexString(std::string_view str) {
    uint64_t value;
    if (!base::HexStringToUint64(str, &value)) return std::nullopt;
    if (value >= uint64_t{GetModulus()}) {
      LOG(ERROR) << "value(" << str << ") is greater than or equal to modulus";
      return std::nullopt;
    }
    return PrimeField(value);
  }

  constexpr static PrimeField FromBigInt(const BigInt<N>& big_int) {
    return PrimeField(big_int);
  }

  constexpr static PrimeField FromMontgomery(const MontgomeryTy& mont) {
    return PrimeField(BigInt<N>::FromMontgomery64(mont, Config::kModulus,
                                                  Config::kInverse64));
  }

  static PrimeField FromMpzClass(const mpz_class& value) {
    BigInt<N> big_int;
    gmp::CopyLimbs(value, big_int.limbs);
    return FromBigInt(big_int);
  }

  static void Init() { VLOG(1) << Config::kName << " initialized"; }

  constexpr value_type value() const { return value_; }

  constexpr bool IsZero() const { return value_ == 0; }

  constexpr bool IsOne() const { return value_ == 1; }

  std::string ToString() const { return base::NumberToString(value_); }

  std::string ToHexString(bool pad_zero = false) const {
    std::string str = base::HexToString(value_);
    if (pad_zero) {
      str = base::ToHexStringWithLeadingZero(str, 8);
    }
    return base::MaybePrepend0x(str);
  }

  mpz_class ToMpzClass() const {
    mpz_class ret;
    gmp::WriteLimbs(ToBigInt().limbs, N, &ret);
    return ret;
  }

  // TODO(chokobole): Support bigendian.
  constexpr BigInt<N> ToBigInt() const { return BigInt<N>(value_); }

  constexpr MontgomeryTy ToMontgomery() const {
    return MontgomeryTy(
        Config::Reduce(uint64_t{value_} * Config::kMontgomeryR[0]));
  }

  constexpr operator uint32_t() const { return value_; }

  uint32_t operator[](size_t i) const {
    DCHECK_EQ(i, size_t{0});
    return value_;
  }

  bool operator==(PrimeField other) const { return value_ == other.value_; }
  bool operator!=(PrimeField other) const { return value_ != other.value_; }
  bool operator<(PrimeField other) const { return value_ < other.value_; }
  bool operator>(PrimeField other) const { return value_ > other.value_; }
  bool operator<=(PrimeField other) const { return value_ <= other.value_; }
  bool operator>=(PrimeField other) const { return value_ >= other.value_; }

  // AdditiveSemigroup methods
  constexpr PrimeField Add(PrimeField other) const {
    return PrimeField(Config::Reduce(uint64_t{value_} + other.value_));
  }

  constexpr PrimeField& AddInPlace(PrimeField other) {
    value_ = Config::Reduce(uint64_t{value_} + other.value_);
    return *this;
  }

  // AdditiveGroup methods
  constexpr PrimeField Sub(PrimeField other) const {
    return PrimeField(
        Config::Reduce(uint64_t{value_} + GetModulus() - other.value_));
  }

  constexpr PrimeField& SubInPlace(PrimeField other) {
    value_ = Config::Reduce(uint64_t{value_} + GetModulus() - other.value_);
    return *this;
  }

  constexpr PrimeField Negate() const {
    return PrimeField(Config::Reduce(uint64_t{GetModulus()} - value_));
  }

  constexpr PrimeField& NegateInPlace() {
    value_ = Config::Reduce(uint64_t{GetModulus()} - value_);
    return *this;
  }

  // MultiplicativeSemigroup methods
  constexpr PrimeField Mul(PrimeField other) const {
// NOTE(chokobole): g++ 11.4.0 gives a warning. It seems to be a g++ bug.
#if defined(COMPILER_GCC) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    return PrimeField(Config::Reduce(uint64_t{value_} * other.value_));
#if defined(COMPILER_GCC) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
  }

  constexpr PrimeField& MulInPlace(PrimeField other) {
    value_ = Config::Reduce(uint64_t{value_} * other.value_);
    return *this;
  }

  // MultiplicativeGroup methods
  constexpr PrimeField Inverse() const {
    // |result.s| * |value_| + |result.t| * |GetModulus()| = |result.r|
    // |result.s| * |value_| = |result.r| (mod |GetModulus()|)
    EGCD<int64_t>::Result result = EGCD<int64_t>::Compute(value_, GetModulus());
    DCHECK_EQ(result.r, 1);
    if (result.s > 0) {
      return PrimeField(Config::Reduce(result.s));
    } else {
      return PrimeField(Config::Reduce(uint64_t{GetModulus()} + result.s));
    }
  }

  constexpr PrimeField& InverseInPlace() {
    // See comment in |Inverse()|.
    EGCD<int64_t>::Result result = EGCD<int64_t>::Compute(value_, GetModulus());
    DCHECK_EQ(result.r, 1);
    if (result.s > 0) {
      value_ = Config::Reduce(result.s);
    } else {
      value_ = Config::Reduce(uint64_t{GetModulus()} + result.s);
    }
    return *this;
  }

 private:
  constexpr static uint32_t GetModulus() {
    return static_cast<uint32_t>(Config::kModulus[0]);
  }

  uint32_t value_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_SMALL_PRIME_FIELD_H_
