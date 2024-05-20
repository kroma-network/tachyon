#ifndef TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_GOLDILOCKS_PRIME_FIELD_X86_SPECIAL_H_
#define TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_GOLDILOCKS_PRIME_FIELD_X86_SPECIAL_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <string>

#include "tachyon/math/finite_fields/goldilocks/goldilocks_config.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::math {

template <typename _Config>
class PrimeField<_Config, std::enable_if_t<_Config::kIsGoldilocks>> final
    : public PrimeFieldBase<PrimeField<_Config>> {
 public:
#if defined(USE_MONTGOMERY)
  static_assert(USE_MONTGOMERY == 0);
#endif  // defined(USE_MONTGOMERY)

  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;

  using Config = _Config;
  using BigIntTy = BigInt<N>;
  using value_type = BigInt<N>;

  PrimeField() = default;
  explicit PrimeField(uint64_t value);
  explicit PrimeField(BigInt<N> value) : PrimeField(value[0]) {}
  PrimeField(const PrimeField& other) = default;
  PrimeField& operator=(const PrimeField& other) = default;
  PrimeField(PrimeField&& other) = default;
  PrimeField& operator=(PrimeField&& other) = default;

  constexpr static PrimeField Zero() { return PrimeField(); }
  static PrimeField One();
  static PrimeField Random();

  static std::optional<PrimeField> FromDecString(std::string_view str);
  static std::optional<PrimeField> FromHexString(std::string_view str);
  static PrimeField FromBigInt(BigInt<N> big_int);
#if USE_MONTGOMERY == 1
  static PrimeField FromMontgomery(BigInt<N> big_int);
#endif

  static void Init() { VLOG(1) << Config::kName << " initialized"; }

  // NOTE(chokobole): To be consistent with `PrimeField<F>` defined in
  // prime_field_fallback.h, it returns the value as a `BigInt<N>`.
  value_type value() const { return value_type(value_); }

  bool IsZero() const;
  bool IsOne() const;

  std::string ToString() const;
  std::string ToHexString(bool pad_zero = false) const;

  // TODO(chokobole): Support bigendian.
  BigInt<N> ToBigInt() const { return BigInt<N>(value_); }

  operator uint64_t() const { return value_; }

  uint64_t operator[](size_t i) const {
    DCHECK_EQ(i, size_t{0});
    return value_;
  }

  bool operator==(const PrimeField& other) const {
    return value_ == other.value_;
  }
  bool operator!=(const PrimeField& other) const {
    return value_ != other.value_;
  }
  bool operator<(const PrimeField& other) const {
    return value_ < other.value_;
  }
  bool operator>(const PrimeField& other) const {
    return value_ > other.value_;
  }
  bool operator<=(const PrimeField& other) const {
    return value_ <= other.value_;
  }
  bool operator>=(const PrimeField& other) const {
    return value_ >= other.value_;
  }

  // AdditiveSemigroup methods
  PrimeField Add(const PrimeField& other) const;
  PrimeField& AddInPlace(const PrimeField& other);

  // AdditiveGroup methods
  PrimeField Sub(const PrimeField& other) const;
  PrimeField& SubInPlace(const PrimeField& other);
  PrimeField Negate() const;
  PrimeField& NegateInPlace();

  // TODO(chokobole): Support bigendian.
  // MultiplicativeSemigroup methods
  PrimeField Mul(const PrimeField& other) const;
  PrimeField& MulInPlace(const PrimeField& other);
  PrimeField SquareImpl() const;
  PrimeField& SquareImplInPlace();

  // MultiplicativeGroup methods
  PrimeField Inverse() const;
  PrimeField& InverseInPlace();

 private:
  uint64_t value_;
};

extern template class PrimeField<GoldilocksConfig>;

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_GOLDILOCKS_PRIME_FIELD_X86_SPECIAL_H_
