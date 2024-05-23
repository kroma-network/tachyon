// Copyright 2023 Ulvetanna Inc.
// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.ulvetanna file.

#ifndef TACHYON_MATH_FINITE_FIELDS_BINARY_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_BINARY_FIELD_H_

#include <stddef.h>
#include <stdint.h>

#include <limits>
#include <optional>
#include <string>
#include <type_traits>

#include "tachyon/base/logging.h"
#include "tachyon/base/random.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/finite_fields/binary_tower_operations.h"
#include "tachyon/math/finite_fields/finite_field.h"

namespace tachyon::math {

template <typename _Config>
class BinaryField final : public FiniteField<BinaryField<_Config>> {
 public:
  constexpr static size_t kBits = _Config::kModulusBits - 1;
  constexpr static size_t kLimbNums = (kBits + 63) / 64;
  constexpr static size_t N = kLimbNums;

  using Config = _Config;
  using Type = uint8_t;
  using BigIntTy = BigInt<N>;
  using value_type = Type;

  using CpuField = BinaryField<Config>;
  using GpuField = BinaryField<Config>;

  constexpr BinaryField() = default;
  constexpr explicit BinaryField(Type value) : value_(value) {
    DCHECK_GE(value_, Type{0});
    DCHECK_LE(value_, GetMax());
  }
  constexpr explicit BinaryField(BigInt<N> value) : BinaryField(value[0]) {
    DCHECK_LE(value[0], GetMax());
  }
  constexpr BinaryField(const BinaryField& other) = default;
  constexpr BinaryField& operator=(const BinaryField& other) = default;
  constexpr BinaryField(BinaryField&& other) = default;
  constexpr BinaryField& operator=(BinaryField&& other) = default;

  constexpr static BinaryField Zero() { return BinaryField(); }
  constexpr static BinaryField One() { return BinaryField(1); }

  static BinaryField Random() {
    return BinaryField(
        base::Uniform(base::Range<Type, true, true>::Until(GetMax())));
  }

  static std::optional<BinaryField> FromDecString(std::string_view str) {
    using MaybePromotedType = std::conditional_t<(kBits <= 32), uint32_t, Type>;
    MaybePromotedType value;
    if (!absl::SimpleAtoi(str, &value)) return std::nullopt;
    if (value > GetMax()) {
      LOG(ERROR) << "value(" << str << ") is greater than or equal to modulus";
      return std::nullopt;
    }
    return BinaryField(value);
  }
  static std::optional<BinaryField> FromHexString(std::string_view str) {
    using MaybePromotedType = std::conditional_t<(kBits <= 32), uint32_t, Type>;
    MaybePromotedType value;
    if (!absl::SimpleHexAtoi(str, &value)) return std::nullopt;
    if (value > GetMax()) {
      LOG(ERROR) << "value(" << str << ") is greater than or equal to modulus";
      return std::nullopt;
    }
    return BinaryField(value);
  }

  constexpr static BinaryField FromBigInt(BigInt<N> big_int) {
    return BinaryField(big_int);
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

  constexpr BigInt<N> ToBigInt() const { return BigInt<N>(value_); }

  bool operator==(BinaryField other) const { return value_ == other.value_; }
  bool operator!=(BinaryField other) const { return value_ != other.value_; }
  bool operator<(BinaryField other) const { return value_ < other.value_; }
  bool operator>(BinaryField other) const { return value_ > other.value_; }
  bool operator<=(BinaryField other) const { return value_ <= other.value_; }
  bool operator>=(BinaryField other) const { return value_ >= other.value_; }

  // AdditiveSemigroup methods
  constexpr BinaryField Add(BinaryField other) const {
    return BinaryField(value_ ^ other.value_);
  }

  constexpr BinaryField& AddInPlace(BinaryField other) {
    value_ ^= other.value_;
    return *this;
  }

  constexpr BinaryField DoubleImpl() const { return BinaryField::Zero(); }

  constexpr BinaryField& DoubleImplInPlace() {
    value_ = 0;
    return *this;
  }

  // AdditiveGroup methods
  constexpr BinaryField Sub(BinaryField other) const {
    return BinaryField(value_ ^ other.value_);
  }

  constexpr BinaryField& SubInPlace(BinaryField other) {
    value_ ^= other.value_;
    return *this;
  }

  constexpr BinaryField Negate() const { return BinaryField(value_); }

  constexpr BinaryField& NegateInPlace() { return *this; }

  // MultiplicativeSemigroup methods
  constexpr BinaryField Mul(BinaryField other) const {
    return BinaryTowerOperations<BinaryField>::Mul(*this, other);
  }

  constexpr BinaryField& MulInPlace(BinaryField other) {
    return *this = BinaryTowerOperations<BinaryField>::Mul(*this, other);
  }

  constexpr BinaryField SquareImpl() const {
    return BinaryTowerOperations<BinaryField>::Square(*this);
  }

  constexpr BinaryField& SquareImplInPlace() {
    return *this = BinaryTowerOperations<BinaryField>::Square(*this);
  }

  // MultiplicativeGroup methods
  constexpr std::optional<BinaryField> Inverse() const {
    return BinaryTowerOperations<BinaryField>::Inverse(*this);
  }

  [[nodiscard]] constexpr std::optional<BinaryField*> InverseInPlace() {
    std::optional<BinaryField> ret =
        BinaryTowerOperations<BinaryField>::Inverse(*this);
    if (LIKELY(ret)) {
      *this = *ret;
      return this;
    }
    return std::nullopt;
  }

 private:
  constexpr static Type GetMax() { return (Type{1} << kBits) - 1; }

  Type value_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_BINARY_FIELD_H_
