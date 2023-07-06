#if defined(TACHYON_GMP_BACKEND)

#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_GMP_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_GMP_H_

#include <stddef.h>

#include <ostream>
#include <string>

#include <gmpxx.h>

#include "tachyon/base/no_destructor.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/base/gmp_util.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon {
namespace math {

template <typename F, size_t _MODULUS_BITS>
class PrimeFieldGmp : public PrimeFieldBase<F> {
 public:
  static constexpr size_t MODULUS_BITS = _MODULUS_BITS;

  using value_type = mpz_class;

  static constexpr size_t LIMB_NUMS = (MODULUS_BITS + 63) / 64;

  PrimeFieldGmp() = default;
  explicit PrimeFieldGmp(const mpz_class& value) : value_(value) {
    Normalize();
  }
  explicit PrimeFieldGmp(mpz_class&& value) : value_(std::move(value)) {
    Normalize();
  }
  PrimeFieldGmp(const PrimeFieldGmp& other) = default;
  PrimeFieldGmp& operator=(const PrimeFieldGmp& other) = default;
  PrimeFieldGmp(PrimeFieldGmp&& other) = default;
  PrimeFieldGmp& operator=(PrimeFieldGmp&& other) = default;

  const value_type& value() const { return value_; }

  static F Zero() { return F(); }

  static F One() { return F(1); }

  static F Random() {
    mpz_class value;
    mpz_urandomm(value.get_mpz_t(), gmp::GetRandomState(),
                 RawModulus().get_mpz_t());
    return F(value);
  }

  static F FromDecString(std::string_view str) {
    mpz_class value;
    gmp::MustParseIntoMpz(str, 10, &value);
    return F(std::move(value));
  }
  static F FromHexString(std::string_view str) {
    mpz_class value;
    gmp::MustParseIntoMpz(str, 16, &value);
    return F(std::move(value));
  }

  bool IsZero() const { return *this == Zero(); }

  bool IsOne() const { return *this == One(); }

  std::string ToString() const { return value_.get_str(); }
  std::string ToHexString() const {
    return base::MaybePrepend0x(value_.get_str(16));
  }

  // TODO(chokobole): Can we avoid copying?
  mpz_class ToMpzClass() const { return value_; }

  [[nodiscard]] constexpr bool ToInt64(int64_t* out) const {
    if (value_.fits_slong_p()) {
      *out = value_.get_si();
      return true;
    }
    return false;
  }

  [[nodiscard]] constexpr bool ToUint64(uint64_t* out) const {
    if (value_.fits_ulong_p()) {
      *out = value_.get_ui();
      return true;
    }
    return false;
  }

  bool operator==(const PrimeFieldGmp& other) const {
    return mpz_cmp(value_.get_mpz_t(), other.value_.get_mpz_t()) == 0;
  }

  bool operator!=(const PrimeFieldGmp& other) const {
    return !operator==(other);
  }

  bool operator<(const PrimeFieldGmp& other) const {
    return mpz_cmp(value_.get_mpz_t(), other.value_.get_mpz_t()) < 0;
  }

  bool operator>(const PrimeFieldGmp& other) const {
    return mpz_cmp(value_.get_mpz_t(), other.value_.get_mpz_t()) > 0;
  }

  bool operator<=(const PrimeFieldGmp& other) const {
    return mpz_cmp(value_.get_mpz_t(), other.value_.get_mpz_t()) <= 0;
  }

  bool operator>=(const PrimeFieldGmp& other) const {
    return mpz_cmp(value_.get_mpz_t(), other.value_.get_mpz_t()) >= 0;
  }

  // AdditiveMonoid methods
  F& DoubleInPlace() { return AddInPlace(static_cast<const F&>(*this)); }

  // AdditiveGroup methods
  F& NegativeInPlace() {
    value_ = RawModulus() - value_;
    return static_cast<F&>(*this);
  }

  // MultiplicativeMonoid methods
  F& SquareInPlace() { return MulInPlace(static_cast<F&>(*this)); }

  // MultiplicativeGroup methods
  F& InverseInPlace() {
    mpz_invert(value_.get_mpz_t(), value_.get_mpz_t(),
               RawModulus().get_mpz_t());
    return static_cast<F&>(*this);
  }

 protected:
  static mpz_class& RawModulus() {
    static base::NoDestructor<mpz_class> prime;
    return *prime;
  }

 private:
  friend struct internal::SupportsAdd<F>;
  friend struct internal::SupportsSub<F>;
  friend struct internal::SupportsMul<F>;
  friend struct internal::SupportsDiv<F>;
  friend class AdditiveMonoid<F>;
  friend class AdditiveGroup<F>;
  friend class MultiplicativeMonoid<F>;
  friend class MultiplicativeGroup<F>;
  friend class PrimeFieldBase<F>;

  // AdditiveMonoid methods
  F Add(const F& other) const { return F(DoMod(value_ + other.value_)); }

  F& AddInPlace(const F& other) {
    value_ = DoMod(value_ + other.value_);
    return static_cast<F&>(*this);
  }

  // AdditiveGroup methods
  F Sub(const F& other) const {
    F ret = F((value_ - other.value_) % RawModulus());
    return ret.Normalize();
  }

  F& SubInPlace(const F& other) {
    value_ = (value_ - other.value_) % RawModulus();
    return Normalize();
  }

  // MultiplicativeMonoid methods
  F Mul(const F& other) const { return F(DoMod(value_ * other.value_)); }

  F& MulInPlace(const F& other) {
    value_ = DoMod(value_ * other.value_);
    return static_cast<F&>(*this);
  }

  // MultiplicativeGroup methods
  F Div(const F& other) const {
    return F(DoMod(value_ * other.Inverse().value_));
  }

  F& DivInPlace(const F& other) {
    value_ = DoMod(value_ * other.Inverse().value_);
    return static_cast<F&>(*this);
  }

  // PrimeFieldBase methods
  // This and `DoMod()` are completely unrelated. `Mod()` exists to support the
  // `operator%()` defined in `PrimeFieldBase`.
  uint64_t Mod(uint64_t mod) const {
    return mpz_fdiv_ui(value_.get_mpz_t(), mod);
  }

  static mpz_class DoMod(mpz_class value) { return value % RawModulus(); }

  F& Normalize() {
    if (gmp::IsNegative(value_)) {
      value_ = RawModulus() + value_;
    }
    return static_cast<F&>(*this);
  }

  mpz_class value_;
};

template <typename F, size_t MODULUS_BITS>
std::ostream& operator<<(std::ostream& os,
                         const PrimeFieldGmp<F, MODULUS_BITS>& f) {
  return os << f.ToString();
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_GMP_H_

#endif  // defined(TACHYON_GMP_BACKEND)
