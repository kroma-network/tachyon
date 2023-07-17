#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_H_

#include <stddef.h>
#include <stdint.h>

#include <ostream>
#include <string>

#include "third_party/gmp/include/gmpxx.h"

#include "tachyon/base/random.h"
#include "tachyon/math/base/big_int.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon {
namespace math {

template <typename _Config>
class PrimeFieldMont : public PrimeFieldBase<PrimeFieldMont<_Config>> {
 public:
  static constexpr size_t kModulusBits = _Config::kModulusBits;
  static constexpr size_t kLimbNums = (kModulusBits + 63) / 64;
  static BigInt<kLimbNums> kModulus;
  static constexpr bool kIsTriviallyDestructible = true;

  using Config = _Config;
  using value_type = BigInt<kLimbNums>;

  constexpr PrimeFieldMont() = default;
  constexpr explicit PrimeFieldMont(int value)
      : PrimeFieldMont(static_cast<uint64_t>(value)) {}
  constexpr explicit PrimeFieldMont(uint64_t value) : value_(value) {
    DCHECK_LT(value_, kModulus);
  }
  constexpr explicit PrimeFieldMont(const BigInt<kLimbNums>& value)
      : value_(value) {
    DCHECK_LT(value_, kModulus);
  }
  constexpr PrimeFieldMont(const PrimeFieldMont& other) = default;
  constexpr PrimeFieldMont& operator=(const PrimeFieldMont& other) = default;
  constexpr PrimeFieldMont(PrimeFieldMont&& other) = default;
  constexpr PrimeFieldMont& operator=(PrimeFieldMont&& other) = default;

  constexpr static PrimeFieldMont Zero() {
    PrimeFieldMont ret;
    ret.value_ = BigInt<kLimbNums>::Zero();
    return ret;
  }

  constexpr static PrimeFieldMont One() {
    PrimeFieldMont ret;
    ret.value_ = BigInt<kLimbNums>::One();
    return ret;
  }

  static PrimeFieldMont Random() {
    NOTIMPLEMENTED();
    return PrimeFieldMont();
  }

  constexpr static PrimeFieldMont FromDecString(std::string_view str) {
    PrimeFieldMont ret;
    ret.value_ = BigInt<kLimbNums>::FromDecString(str);
    return ret;
  }
  constexpr static PrimeFieldMont FromHexString(std::string_view str) {
    PrimeFieldMont ret;
    ret.value_ = BigInt<kLimbNums>::FromHexString(str);
    return ret;
  }

  constexpr static void Init() {
    for (size_t i = 0; i < kLimbNums; ++i) {
      kModulus[i] = Config::kModulus[i];
    }
  }

  const value_type& value() const { return value_; }

  constexpr bool IsZero() const { return value_.IsZero(); }

  constexpr bool IsOne() const { return value_.IsOne(); }

  std::string ToString() const { return value_.ToString(); }
  std::string ToHexString() const { return value_.ToHexString(); }

  mpz_class ToMpzClass() const {
    NOTIMPLEMENTED();
    return {};
  }

  const BigInt<kLimbNums>& ToBigInt() const { return value_; }

  constexpr bool operator==(const PrimeFieldMont& other) const {
    return value_ == other.value_;
  }

  constexpr bool operator!=(const PrimeFieldMont& other) const {
    return value_ != other.value_;
  }

  constexpr bool operator<(const PrimeFieldMont& other) const {
    return value_ < other.value_;
  }

  constexpr bool operator>(const PrimeFieldMont& other) const {
    return value_ > other.value_;
  }

  constexpr bool operator<=(const PrimeFieldMont& other) const {
    return value_ <= other.value_;
  }

  constexpr bool operator>=(const PrimeFieldMont& other) const {
    return value_ >= other.value_;
  }

  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  mpz_class DivBy2Exp(uint64_t exp) const {
    mpz_class ret;
    NOTIMPLEMENTED();
    return ret;
  }

  // AdditiveMonoid methods
  PrimeFieldMont& AddInPlace(const PrimeFieldMont& other) {
    NOTIMPLEMENTED();
    return *this;
  }

  PrimeFieldMont& DoubleInPlace() {
    NOTIMPLEMENTED();
    return *this;
  }

  // AdditiveGroup methods
  PrimeFieldMont& SubInPlace(const PrimeFieldMont& other) {
    NOTIMPLEMENTED();
    return *this;
  }

  PrimeFieldMont& NegInPlace() {
    NOTIMPLEMENTED();
    return *this;
  }

  // MultiplicativeMonoid methods
  PrimeFieldMont& MulInPlace(const PrimeFieldMont& other) {
    NOTIMPLEMENTED();
    return *this;
  }

  PrimeFieldMont& SquareInPlace() { return MulInPlace(*this); }

  // MultiplicativeGroup methods
  PrimeFieldMont& InverseInPlace() {
    CHECK(!IsZero());
    return *this;
  }

 private:
  BigInt<kLimbNums> value_;
};

// static
template <typename Config>
BigInt<PrimeFieldMont<Config>::kLimbNums> PrimeFieldMont<Config>::kModulus;

template <typename Config>
std::ostream& operator<<(std::ostream& os, const PrimeFieldMont<Config>& f) {
  return os << f.ToString();
}

template <typename Config>
class MultiplicativeIdentity<PrimeFieldMont<Config>> {
 public:
  using F = PrimeFieldMont<Config>;

  static const F& One() {
    static F one(F::One());
    return one;
  }

  constexpr static bool IsOne(const F& value) { return value.IsOne(); }
};

template <typename Config>
class AdditiveIdentity<PrimeFieldMont<Config>> {
 public:
  using F = PrimeFieldMont<Config>;

  static const F& Zero() {
    static F zero(F::Zero());
    return zero;
  }

  constexpr static bool IsZero(const F& value) { return value.IsZero(); }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_MONT_H_
