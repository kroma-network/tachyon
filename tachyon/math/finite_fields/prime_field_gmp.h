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
#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon::math {

template <typename Config>
class PrimeFieldGpu;

template <typename _Config>
class PrimeFieldGmp : public PrimeFieldBase<PrimeFieldGmp<_Config>> {
 public:
  static_assert(GMP_LIMB_BITS == 64, "This code assumes limb bits is 64 bit");
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;

  using Config = _Config;
  using BigIntTy = BigInt<N>;
  using value_type = mpz_class;

  using CpuField = PrimeFieldGmp<Config>;
  using GpuField = PrimeFieldGpu<Config>;

  PrimeFieldGmp() = default;
  explicit PrimeFieldGmp(const mpz_class& value) : value_(value) {
    DCHECK(!gmp::IsNegative(value_));
    DCHECK_LT(value_, Modulus());
  }
  explicit PrimeFieldGmp(mpz_class&& value) : value_(std::move(value)) {
    DCHECK(!gmp::IsNegative(value_));
    DCHECK_LT(value_, Modulus());
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

  static PrimeFieldGmp FromBigInt(const BigInt<N>& big_int) {
    mpz_class out;
    gmp::WriteLimbs(big_int.limbs, N, &out);
    return PrimeFieldGmp(std::move(out));
  }

  static PrimeFieldGmp FromMontgomery(const BigInt<N>& big_int) {
    return FromBigInt(BigInt<N>::FromMontgomery64(big_int, Config::kModulus,
                                                  Config::kInverse64));
  }

  static void Init() {
    static absl::once_flag once;
    absl::call_once(once, []() {
#if ARCH_CPU_BIG_ENDIAN
      uint64_t modulus[N];
      for (size_t i = 0; i < N; ++i) {
        uint64_t value = absl::little_endian::Load64(Config::kModulus.limbs[i]);
        memcpy(&modulus[N - i - 1], &value, sizeof(uint64_t));
      }
#else
          const uint64_t* modulus = Config::kModulus.limbs;
#endif
      gmp::WriteLimbs(modulus, N, &Modulus());
    });
  }

  static mpz_class& Modulus() {
    static base::NoDestructor<mpz_class> modulus;
    return *modulus;
  }

  const value_type& value() const { return value_; }
  size_t GetLimbSize() const { return gmp::GetLimbSize(value_); }

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

  BigInt<N> ToMontgomery() const {
    return PrimeField<Config>(ToBigInt()).value();
  }

  const uint64_t& operator[](size_t i) const {
    return gmp::GetLimbConstRef(value_, i);
  }
  uint64_t& operator[](size_t i) { return gmp::GetLimbRef(value_, i); }

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
  BigInt<N> DivBy2Exp(uint32_t exp) const {
    return ToBigInt().DivBy2ExpInPlace(exp);
  }

  // AdditiveSemigroup methods
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
    mpz_class sub = value_ - other.value_;
    if (sub < 0) {
      sub += Modulus();
    }
    return PrimeFieldGmp(sub);
  }

  PrimeFieldGmp& SubInPlace(const PrimeFieldGmp& other) {
    value_ -= other.value_;
    if (value_ < 0) {
      value_ += Modulus();
    }
    return *this;
  }

  PrimeFieldGmp& NegInPlace() {
    if (value_ == mpz_class(0)) return *this;
    value_ = Modulus() - value_;
    return *this;
  }

  // MultiplicativeSemigroup methods
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

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_GMP_H_

#endif  // defined(TACHYON_GMP_BACKEND)
