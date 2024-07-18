// Copyright 2023 Ulvetanna Inc.
// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.ulvetanna file.

#ifndef TACHYON_MATH_FINITE_FIELDS_BINARY_FIELDS_BINARY_FIELD_H_
#define TACHYON_MATH_FINITE_FIELDS_BINARY_FIELDS_BINARY_FIELD_H_

#include <stddef.h>
#include <stdint.h>

#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>

#include "tachyon/base/logging.h"
#include "tachyon/base/numerics/safe_conversions.h"
#include "tachyon/base/random.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/finite_fields/binary_fields/binary_field_traits_forward.h"
#include "tachyon/math/finite_fields/binary_fields/binary_tower_operations.h"
#include "tachyon/math/finite_fields/finite_field.h"

namespace tachyon::math {

template <typename _Config>
class BinaryField final : public FiniteField<BinaryField<_Config>> {
 public:
  constexpr static size_t kBits = _Config::kModulusBits - 1;
  constexpr static size_t kLimbNums = (kBits + 63) / 64;
  constexpr static size_t N = kLimbNums;

  using Config = _Config;
  using Type = std::conditional_t<
      kBits <= 8, uint8_t,
      std::conditional_t<
          kBits == 16, uint16_t,
          std::conditional_t<
              kBits == 32, uint32_t,
              std::conditional_t<kBits == 64, uint64_t, BigInt<2>>>>>;
  using SubField = BinaryField<typename BinaryFieldTraits<Config>::SubConfig>;
  using BigIntTy = BigInt<N>;
  using value_type = Type;

  using CpuField = BinaryField<Config>;
  using GpuField = BinaryField<Config>;

  constexpr BinaryField() = default;
  template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
  constexpr explicit BinaryField(T value) {
    if constexpr (kBits <= 64) {
      DCHECK(base::IsValueInRangeForNumericType<Type>(value));
      DCHECK_LE(static_cast<Type>(value), GetMax());
      value_ = static_cast<Type>(value);
    } else {
      value_[0] = value;
    }
  }
  constexpr explicit BinaryField(BigInt<1> value) : BinaryField(value[0]) {
    static_assert(kBits <= 64);
  }
  constexpr explicit BinaryField(BigInt<2> value) : value_(value) {
    static_assert(kBits == 128);
  }
  constexpr BinaryField(const BinaryField& other) = default;
  constexpr BinaryField& operator=(const BinaryField& other) = default;
  constexpr BinaryField(BinaryField&& other) = default;
  constexpr BinaryField& operator=(BinaryField&& other) = default;

  constexpr static BinaryField Zero() { return BinaryField(); }
  constexpr static BinaryField One() { return BinaryField(1); }
  constexpr static BinaryField MinusOne() { return One(); }

  static BinaryField Random() {
    if constexpr (kBits <= 64) {
      return BinaryField(
          base::Uniform(base::Range<Type, true, true>::Until(GetMax())));
    } else {
      static_assert(kBits == 128);
      return BinaryField(BigInt<2>::Random());
    }
  }

  static std::optional<BinaryField> FromDecString(std::string_view str) {
    if constexpr (kBits <= 64) {
      using MaybePromotedType =
          std::conditional_t<(kBits <= 32), uint32_t, Type>;
      MaybePromotedType value;
      if (!absl::SimpleAtoi(str, &value)) return std::nullopt;
      if constexpr (kBits <= 16) {
        if (value > GetMax()) {
          LOG(ERROR) << "value(" << str
                     << ") is greater than or equal to modulus";
          return std::nullopt;
        }
      }
      return BinaryField(value);
    } else {
      static_assert(kBits == 128);
      std::optional<BigInt<2>> value = BigInt<2>::FromDecString(str);
      if (!value) return std::nullopt;
      return BinaryField(*value);
    }
  }
  static std::optional<BinaryField> FromHexString(std::string_view str) {
    if constexpr (kBits <= 64) {
      using MaybePromotedType =
          std::conditional_t<(kBits <= 32), uint32_t, Type>;
      MaybePromotedType value;
      if (!absl::SimpleHexAtoi(str, &value)) return std::nullopt;
      if constexpr (kBits <= 16) {
        if (value > GetMax()) {
          LOG(ERROR) << "value(" << str
                     << ") is greater than or equal to modulus";
          return std::nullopt;
        }
      }
      return BinaryField(value);
    } else {
      static_assert(kBits == 128);
      std::optional<BigInt<2>> value = BigInt<2>::FromHexString(str);
      if (!value) return std::nullopt;
      return BinaryField(*value);
    }
  }

  constexpr static BinaryField FromBigInt(BigInt<N> big_int) {
    return BinaryField(big_int);
  }

  template <typename T = Type,
            std::enable_if_t<!std::is_same_v<T, uint8_t>>* = nullptr>
  constexpr static BinaryField Compose(SubField lo, SubField hi) {
    if constexpr (kBits <= 64) {
      using sub_value_type = typename SubField::value_type;
      return BinaryField(
          (value_type{hi.value()} << (sizeof(sub_value_type) * 8)) +
          value_type{lo.value()});
    } else {
      static_assert(kBits == 128);
      return BinaryField(BigInt<2>{lo.value(), hi.value()});
    }
  }

  static void Init() { VLOG(1) << Config::kName << " initialized"; }

  constexpr value_type value() const { return value_; }

  constexpr bool IsZero() const {
    if constexpr (kBits <= 64) {
      return value_ == 0;
    } else {
      static_assert(kBits == 128);
      return value_.IsZero();
    }
  }

  constexpr bool IsOne() const {
    if constexpr (kBits <= 64) {
      return value_ == 1;
    } else {
      static_assert(kBits == 128);
      return value_.IsOne();
    }
  }

  constexpr bool IsMinusOne() const { return IsOne(); }

  template <typename T = Type,
            std::enable_if_t<!std::is_same_v<T, uint8_t>>* = nullptr>
  constexpr std::tuple<SubField, SubField> Decompose() const {
    std::tuple<SubField, SubField> ret;
    if constexpr (kBits <= 64) {
      using sub_value_type = typename SubField::value_type;
      std::get<0>(ret) = SubField(static_cast<sub_value_type>(value_));
      std::get<1>(ret) = SubField(
          static_cast<sub_value_type>(value_ >> (sizeof(sub_value_type) * 8)));
    } else {
      static_assert(kBits == 128);
      std::get<0>(ret) = SubField(value_[0]);
      std::get<1>(ret) = SubField(value_[1]);
    }
    return ret;
  }

  std::string ToString() const {
    if constexpr (kBits <= 64) {
      return base::NumberToString(value_);
    } else {
      static_assert(kBits == 128);
      return ToBigInt().ToString();
    }
  }

  std::string ToHexString(bool pad_zero = false) const {
    if constexpr (kBits <= 64) {
      std::string str = base::HexToString(value_);
      if (pad_zero) {
        str = base::ToHexStringWithLeadingZero(str, 8);
      }
      return base::MaybePrepend0x(str);
    } else {
      static_assert(kBits == 128);
      return ToBigInt().ToHexString(pad_zero);
    }
  }

  constexpr BigInt<N> ToBigInt() const {
    if constexpr (kBits <= 64) {
      return BigInt<1>(value_);
    } else {
      static_assert(kBits == 128);
      return value_;
    }
  }

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
    value_ = Type{0};
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
  constexpr static Type GetMax() {
    if constexpr (kBits <= 8) {
      return (Type{1} << kBits) - 1;
    } else if constexpr (kBits <= 64) {
      return static_cast<Type>(std::numeric_limits<Type>::max());
    } else {
      static_assert(kBits == 128);
      return BigInt<2>::Max();
    }
  }

  Type value_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_BINARY_FIELDS_BINARY_FIELD_H_
