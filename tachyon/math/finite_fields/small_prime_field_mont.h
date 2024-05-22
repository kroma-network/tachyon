// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_SMALL_PRIME_FIELD_MONT_H_
#define TACHYON_MATH_FINITE_FIELDS_SMALL_PRIME_FIELD_MONT_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <string>

#include "tachyon/base/logging.h"
#include "tachyon/base/random.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::math {

template <typename Config>
class PrimeFieldGpu;

// A prime field is finite field GF(p) where p is a prime number.
template <typename _Config>
class PrimeField<_Config, std::enable_if_t<!_Config::kIsSpecialPrime &&
                                           (_Config::kModulusBits <= 32) &&
                                           _Config::kUseMontgomery>>
    final : public PrimeFieldBase<PrimeField<_Config>> {
 public:
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;

  using Config = _Config;
  using BigIntTy = BigInt<N>;
  using value_type = uint32_t;

  using CpuField = PrimeField<Config>;
  using GpuField = PrimeFieldGpu<Config>;

  constexpr PrimeField() = default;
  constexpr explicit PrimeField(uint32_t value)
      : value_(Config::ToMontgomery(value)) {
    DCHECK_LT(value, GetModulus());
  }
  constexpr explicit PrimeField(BigInt<N> value) : PrimeField(value[0]) {
    DCHECK_LT(value[0], GetModulus());
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

  constexpr static PrimeField FromBigInt(BigInt<N> big_int) {
    return PrimeField(big_int);
  }

  constexpr static PrimeField FromMontgomery(const uint32_t value) {
    PrimeField ret;
    ret.value_ = value;
    return ret;
  }

  static void Init() { VLOG(1) << Config::kName << " initialized"; }

  constexpr value_type value() const { return value_; }

  constexpr bool IsZero() const { return value_ == 0; }

  constexpr bool IsOne() const { return value_ == Config::kOne; }

  std::string ToString() const {
    return base::NumberToString(Config::FromMontgomery(value_));
  }

  std::string ToHexString(bool pad_zero = false) const {
    std::string str = base::HexToString(Config::FromMontgomery(value_));
    if (pad_zero) {
      str = base::ToHexStringWithLeadingZero(str, 8);
    }
    return base::MaybePrepend0x(str);
  }

  // TODO(chokobole): Support bigendian.
  constexpr BigInt<N> ToBigInt() const {
    return BigInt<N>(Config::FromMontgomery(value_));
  }

  constexpr operator uint32_t() const { return value_; }

  constexpr uint32_t operator[](size_t i) const {
    DCHECK_EQ(i, size_t{0});
    return value_;
  }

  constexpr bool operator==(PrimeField other) const {
    return value_ == other.value_;
  }
  constexpr bool operator!=(PrimeField other) const {
    return value_ != other.value_;
  }
  constexpr bool operator<(PrimeField other) const {
    return Config::FromMontgomery(value_) <
           Config::FromMontgomery(other.value_);
  }
  constexpr bool operator>(PrimeField other) const {
    return Config::FromMontgomery(value_) >
           Config::FromMontgomery(other.value_);
  }
  constexpr bool operator<=(PrimeField other) const {
    return Config::FromMontgomery(value_) <=
           Config::FromMontgomery(other.value_);
  }
  constexpr bool operator>=(PrimeField other) const {
    return Config::FromMontgomery(value_) >=
           Config::FromMontgomery(other.value_);
  }

  // AdditiveSemigroup methods
  constexpr PrimeField Add(PrimeField other) const {
    PrimeField ret;
    ret.value_ = Config::AddMod(value_, other.value_);
    return ret;
  }

  constexpr PrimeField& AddInPlace(PrimeField other) {
    value_ = Config::AddMod(value_, other.value_);
    return *this;
  }

  // AdditiveGroup methods
  constexpr PrimeField Sub(PrimeField other) const {
    PrimeField ret;
    ret.value_ = Config::SubMod(value_, other.value_);
    return ret;
  }

  constexpr PrimeField& SubInPlace(PrimeField other) {
    value_ = Config::SubMod(value_, other.value_);
    return *this;
  }

  constexpr PrimeField Negate() const {
    PrimeField ret;
    ret.value_ = Config::SubMod(0, value_);
    return ret;
  }

  constexpr PrimeField& NegateInPlace() {
    value_ = Config::SubMod(0, value_);
    return *this;
  }

  // MultiplicativeSemigroup methods
  constexpr PrimeField Mul(PrimeField other) const {
    PrimeField ret;
    ret.value_ = Config::FromMontgomery(uint64_t{value_} * other.value_);
    return ret;
  }

  constexpr PrimeField& MulInPlace(PrimeField other) {
    value_ = Config::FromMontgomery(uint64_t{value_} * other.value_);
    return *this;
  }

  // MultiplicativeGroup methods
  constexpr PrimeField Inverse() const { return this->Pow(GetModulus() - 2); }

  constexpr PrimeField& InverseInPlace() {
    return *this = this->Pow(GetModulus() - 2);
  }

 private:
  constexpr static uint32_t GetModulus() { return Config::kModulus; }

  uint32_t value_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_SMALL_PRIME_FIELD_MONT_H_
