#if defined(TACHYON_GMP_BACKEND)

#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_GMP_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_GMP_H_

#include <stddef.h>

#include <ostream>
#include <string>

#include "absl/base/call_once.h"
#include "absl/base/internal/endian.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/no_destructor.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon {
namespace math {

template <typename _Config>
class PrimeFieldGmp : public PrimeFieldBase<PrimeFieldGmp<_Config>> {
 public:
  static_assert(GMP_LIMB_BITS == 64, "This code assumes limb bits is 64 bit");
  static constexpr size_t kModulusBits = _Config::kModulusBits;
  static constexpr size_t kLimbNums = (kModulusBits + 63) / 64;
  static constexpr size_t N = kLimbNums;
  static constexpr const uint64_t* kModulus = _Config::kModulus.limbs;
  static constexpr bool kIsTriviallyDestructible = false;

  using Config = _Config;
  using value_type = mpz_class;

  PrimeFieldGmp() = default;
  explicit PrimeFieldGmp(const mpz_class& value) : value_(value) {
    DCHECK(!gmp::IsNegative(value));
    DCHECK_LT(value, Modulus());
  }
  explicit PrimeFieldGmp(mpz_class&& value) : value_(std::move(value)) {
    DCHECK(!gmp::IsNegative(value));
    DCHECK_LT(value, Modulus());
  }
  PrimeFieldGmp(const PrimeFieldGmp& other) = default;
  PrimeFieldGmp& operator=(const PrimeFieldGmp& other) = default;
  PrimeFieldGmp(PrimeFieldGmp&& other) = default;
  PrimeFieldGmp& operator=(PrimeFieldGmp&& other) = default;

  static PrimeFieldGmp Zero() { return PrimeFieldGmp(); }

  static PrimeFieldGmp One() { return PrimeFieldGmp(1); }

  static PrimeFieldGmp Random() {
    return PrimeFieldGmp(gmp::Random(Modulus()));
  }

  static PrimeFieldGmp FromMpzClass(const mpz_class& value) {
    return PrimeFieldGmp(value);
  }
  static PrimeFieldGmp FromMpzClass(mpz_class&& value) {
    return PrimeFieldGmp(std::move(value));
  }

  static PrimeFieldGmp FromDecString(std::string_view str) {
    return PrimeFieldGmp(gmp::FromDecString(str));
  }
  static PrimeFieldGmp FromHexString(std::string_view str) {
    return PrimeFieldGmp(gmp::FromHexString(str));
  }

  static void Init() {
    static absl::once_flag once;
    absl::call_once(once, []() {
#if ARCH_CPU_BIG_ENDIAN
      uint64_t modulus[N];
      for (size_t i = 0; i < N; ++i) {
        uint64_t value = absl::little_endian::Load64(kModulus[i]);
        memcpy(&modulus[N - i - 1], &value, sizeof(uint64_t));
      }
#else
      const uint64_t* modulus = kModulus;
#endif
      gmp::WriteLimbs(modulus, N, &Modulus());
    });
  }

  static mpz_class& Modulus() {
    static base::NoDestructor<mpz_class> modulus;
    return *modulus;
  }

  const value_type& value() const { return value_; }

  bool IsZero() const { return *this == Zero(); }

  bool IsOne() const { return *this == One(); }

  std::string ToString() const { return value_.get_str(); }
  std::string ToHexString() const {
    return base::MaybePrepend0x(value_.get_str(16));
  }

  const mpz_class& ToMpzClass() const { return value_; }

  BigInt<N> ToBigInt() const {
    BigInt<N> result;
    gmp::CopyLimbs(value_, result.limbs);
    return result;
  }

  bool operator==(const PrimeFieldGmp& other) const {
    return value_ == other.value_;
  }

  bool operator!=(const PrimeFieldGmp& other) const {
    return value_ != other.value_;
  }

  bool operator<(const PrimeFieldGmp& other) const {
    return value_ < other.value_;
  }

  bool operator>(const PrimeFieldGmp& other) const {
    return value_ > other.value_;
  }

  bool operator<=(const PrimeFieldGmp& other) const {
    return value_ <= other.value_;
  }

  bool operator>=(const PrimeFieldGmp& other) const {
    return value_ >= other.value_;
  }

  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  mpz_class DivBy2Exp(uint64_t exp) const {
    return gmp::DivBy2Exp(value_, exp);
  }

  // AdditiveMonoid methods
  PrimeFieldGmp Add(const PrimeFieldGmp& other) const {
    return PrimeFieldGmp(DoMod(value_ + other.value_));
  }

  PrimeFieldGmp& AddInPlace(const PrimeFieldGmp& other) {
    value_ = DoMod(value_ + other.value_);
    return *this;
  }

  PrimeFieldGmp& DoubleInPlace() {
    return AddInPlace(static_cast<const PrimeFieldGmp&>(*this));
  }

  // AdditiveGroup methods
  PrimeFieldGmp Sub(const PrimeFieldGmp& other) const {
    PrimeFieldGmp ret = PrimeFieldGmp((value_ - other.value_) % Modulus());
    return ret.Normalize();
  }

  PrimeFieldGmp& SubInPlace(const PrimeFieldGmp& other) {
    value_ = (value_ - other.value_) % Modulus();
    return Normalize();
  }

  PrimeFieldGmp& NegInPlace() {
    if (value_ == mpz_class(0)) return *this;
    value_ = Modulus() - value_;
    return *this;
  }

  // MultiplicativeMonoid methods
  PrimeFieldGmp Mul(const PrimeFieldGmp& other) const {
    return PrimeFieldGmp(DoMod(value_ * other.value_));
  }

  PrimeFieldGmp& MulInPlace(const PrimeFieldGmp& other) {
    value_ = DoMod(value_ * other.value_);
    return *this;
  }

  PrimeFieldGmp& SquareInPlace() {
    return MulInPlace(static_cast<PrimeFieldGmp&>(*this));
  }

  // MultiplicativeGroup methods
  PrimeFieldGmp Div(const PrimeFieldGmp& other) const {
    return PrimeFieldGmp(DoMod(value_ * other.Inverse().value_));
  }

  PrimeFieldGmp& DivInPlace(const PrimeFieldGmp& other) {
    value_ = DoMod(value_ * other.Inverse().value_);
    return *this;
  }

  PrimeFieldGmp& InverseInPlace() {
    mpz_invert(value_.get_mpz_t(), value_.get_mpz_t(), Modulus().get_mpz_t());
    return *this;
  }

 private:
  static mpz_class DoMod(mpz_class value) { return value % Modulus(); }

  PrimeFieldGmp& Normalize() {
    if (gmp::IsNegative(value_)) {
      value_ = Modulus() + value_;
    }
    return *this;
  }

  mpz_class value_;
};

template <typename Config>
std::ostream& operator<<(std::ostream& os, const PrimeFieldGmp<Config>& f) {
  return os << f.ToString();
}

template <typename Config>
class MultiplicativeIdentity<PrimeFieldGmp<Config>> {
 public:
  using F = PrimeFieldGmp<Config>;

  static const F& One() {
    static base::NoDestructor<F> one(F::One());
    return *one;
  }

  constexpr static bool IsOne(const F& value) { return value.IsOne(); }
};

template <typename Config>
class AdditiveIdentity<PrimeFieldGmp<Config>> {
 public:
  using F = PrimeFieldGmp<Config>;

  static const F& Zero() {
    static base::NoDestructor<F> zero(F::Zero());
    return *zero;
  }

  constexpr static bool IsZero(const F& value) { return value.IsZero(); }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_GMP_H_

#endif  // defined(TACHYON_GMP_BACKEND)
