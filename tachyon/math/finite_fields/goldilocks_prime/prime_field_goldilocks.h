#ifndef TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_PRIME_FIELD_GOLDILOCKS_H_
#define TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_PRIME_FIELD_GOLDILOCKS_H_

#include <stddef.h>
#include <stdint.h>

#include <ostream>
#include <string>

#include "third_party/polygon_zkevm_goldilocks/include/goldilocks_base_field.hpp"

#include "tachyon/base/random.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon::math {

template <typename _Config>
class PrimeField<_Config, std::enable_if_t<_Config::kIsGoldilocks>> final
    : public PrimeFieldBase<PrimeField<_Config>> {
 public:
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;

  using Config = _Config;
  using BigIntTy = BigInt<N>;
  using value_type = Goldilocks::Element;

  constexpr PrimeField() = default;
  constexpr explicit PrimeField(uint64_t value)
      : PrimeField(Goldilocks::fromU64(value)) {}
  constexpr explicit PrimeField(int64_t value)
      : PrimeField(Goldilocks::fromS64(value)) {}
  constexpr explicit PrimeField(int32_t value)
      : PrimeField(Goldilocks::fromS32(value)) {}
  constexpr explicit PrimeField(const Goldilocks::Element& value)
      : value_(value) {}
  constexpr PrimeField(const PrimeField& other) = default;
  constexpr PrimeField& operator=(const PrimeField& other) = default;
  constexpr PrimeField(PrimeField&& other) = default;
  constexpr PrimeField& operator=(PrimeField&& other) = default;

  constexpr static PrimeField Zero() { return PrimeField(Goldilocks::zero()); }

  constexpr static PrimeField One() { return PrimeField(Goldilocks::one()); }

  static PrimeField Random() { return PrimeField(RandomForTesting()); }

  static uint64_t RandomForTesting() {
    return base::Uniform(base::Range<uint64_t>::Until(Config::kModulus[0]));
  }

  constexpr static PrimeField FromDecString(std::string_view str) {
    uint64_t value = 0;
    CHECK(base::StringToUint64(str, &value));
    return PrimeField(Goldilocks::fromU64(value));
  }
  constexpr static PrimeField FromHexString(std::string_view str) {
    uint64_t value = 0;
    CHECK(base::HexStringToUint64(str, &value));
    return PrimeField(Goldilocks::fromU64(value));
  }

  constexpr static PrimeField FromBigInt(const BigInt<N>& big_int) {
    return PrimeField(Goldilocks::fromU64(big_int[0]));
  }

  static PrimeField FromMontgomery(const BigInt<N>& big_int) {
    return PrimeField(Goldilocks::from_montgomery(big_int[0]));
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

  constexpr bool IsZero() const { return Goldilocks::isZero(value_); }

  constexpr bool IsOne() const { return Goldilocks::isOne(value_); }

  std::string ToString() const { return Goldilocks::toString(value_, 10); }
  std::string ToHexString(bool pad_zero = false) const {
    std::string str = Goldilocks::toString(value_, 16);
    if (pad_zero) {
      str = base::ToHexStringWithLeadingZero(str);
    }
    return base::MaybePrepend0x(str);
  }

  mpz_class ToMpzClass() const {
    mpz_class ret;
    uint64_t limbs[] = {Goldilocks::toU64(value_)};
    gmp::WriteLimbs(limbs, N, &ret);
    return ret;
  }

  // TODO(chokobole): Support bigendian.
  constexpr BigInt<N> ToBigInt() const {
    return BigInt<N>(Goldilocks::toU64(value_));
  }

  constexpr BigInt<N> ToMontgomery() const {
    return BigInt<N>(Goldilocks::to_montgomery(Goldilocks::toU64(value_)));
  }

  operator int64_t() const { return Goldilocks::toU64(value_); }

  constexpr uint64_t operator[](size_t i) const {
    DCHECK_EQ(i, 0);
    return Goldilocks::toU64(value_);
  }

  constexpr bool operator==(const PrimeField& other) const {
    return Goldilocks::toU64(value_) == Goldilocks::toU64(other.value_);
  }

  constexpr bool operator!=(const PrimeField& other) const {
    return Goldilocks::toU64(value_) != Goldilocks::toU64(other.value_);
  }

  constexpr bool operator<(const PrimeField& other) const {
    return Goldilocks::toU64(value_) < Goldilocks::toU64(other.value_);
  }

  constexpr bool operator>(const PrimeField& other) const {
    return Goldilocks::toU64(value_) > Goldilocks::toU64(other.value_);
  }

  constexpr bool operator<=(const PrimeField& other) const {
    return Goldilocks::toU64(value_) <= Goldilocks::toU64(other.value_);
  }

  constexpr bool operator>=(const PrimeField& other) const {
    return Goldilocks::toU64(value_) >= Goldilocks::toU64(other.value_);
  }

  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  mpz_class DivBy2Exp(uint64_t exp) const {
    return gmp::DivBy2Exp(ToMpzClass(), exp);
  }

  // AdditiveSemigroup methods
  constexpr PrimeField& AddInPlace(const PrimeField& other) {
    Goldilocks::add(value_, value_, other.value_);
    return *this;
  }

  constexpr PrimeField& DoubleInPlace() {
    Goldilocks::add(value_, value_, value_);
    return *this;
  }

  // AdditiveGroup methods
  constexpr PrimeField& SubInPlace(const PrimeField& other) {
    Goldilocks::sub(value_, value_, other.value_);
    return *this;
  }

  constexpr PrimeField& NegInPlace() {
    Goldilocks::neg(value_, value_);
    return *this;
  }

  // TODO(chokobole): Support bigendian.
  // MultiplicativeSemigroup methods
  constexpr PrimeField& MulInPlace(const PrimeField& other) {
    Goldilocks::mul(value_, value_, other.value_);
    return *this;
  }

  constexpr PrimeField& DivInPlace(const PrimeField& other) {
    Goldilocks::div(value_, value_, other.value_);
    return *this;
  }

  constexpr PrimeField& SquareInPlace() {
    Goldilocks::square(value_, value_);
    return *this;
  }

  // MultiplicativeGroup methods
  constexpr PrimeField& InverseInPlace() {
    // See https://github.com/kroma-network/tachyon/issues/76
    CHECK(!IsZero());
    Goldilocks::inv(value_, value_);
    return *this;
  }

 private:
  Goldilocks::Element value_;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_GOLDILOCKS_PRIME_PRIME_FIELD_GOLDILOCKS_H_
