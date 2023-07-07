#ifndef TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_FALLBACK_H_
#define TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_FALLBACK_H_

#include <stddef.h>
#include <stdint.h>

#include <ostream>
#include <string>

#include "absl/strings/str_join.h"
#include "gmpxx.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon {
namespace math {

template <typename F, size_t _MODULUS_BITS>
class PrimeFieldFallback : public PrimeFieldBase<F> {
 public:
  static constexpr size_t MODULUS_BITS = _MODULUS_BITS;
  static constexpr size_t LIMB_NUMS = (MODULUS_BITS + 63) / 64;

  constexpr PrimeFieldFallback() = default;
  constexpr explicit PrimeFieldFallback(uint64_t value) { limbs_[0] = value; }
  constexpr explicit PrimeFieldFallback(uint64_t limbs[LIMB_NUMS])
      : limbs_(limbs) {}
  constexpr PrimeFieldFallback(const PrimeFieldFallback& other) = default;
  constexpr PrimeFieldFallback& operator=(const PrimeFieldFallback& other) =
      default;
  constexpr PrimeFieldFallback(PrimeFieldFallback&& other) = default;
  constexpr PrimeFieldFallback& operator=(PrimeFieldFallback&& other) = default;

  constexpr static F Zero() { return F(); }

  constexpr static F One() { return F(1); }

  constexpr static F FromDecString(std::string_view str) {
    NOTIMPLEMENTED();
    return F();
  }
  constexpr static F FromHexString(std::string_view str) {
    NOTIMPLEMENTED();
    return F();
  }

  constexpr bool IsZero() const { return *this == Zero(); }

  constexpr bool IsOne() const { return *this == One(); }

  std::string ToString() const { return absl::StrJoin(limbs_, ", "); }
  std::string ToHexString() const {
    return MaybePrepend0x(absl::StrJoin(
        base::Reversed(limbs_), "", [](std::string* out, uint64_t limb) {
          if (limb != 0) absl::StrAppend(out, absl::Hex(limb));
        }));
  }

  // TODO(chokobole): Can we avoid copying?
  mpz_class ToMpzClass() const {
    NOTIMPLEMENTED();
    return {};
  }

  constexpr bool ToInt64(int64_t* out) const {
    for (size_t i = 1; i < LIMB_NUMS; ++i) {
      if (limbs_[i] != 0) return false;
    }
    *out = static_cast<int64_t>(limbs_[0]);
    return true;
  }

  constexpr bool ToUint64(uint64_t* out) const {
    for (size_t i = 1; i < LIMB_NUMS; ++i) {
      if (limbs_[i] != 0) return false;
    }
    *out = static_cast<uint64_t>(limbs_[0]);
    return true;
  }

  constexpr bool operator==(const PrimeFieldFallback& other) const {
    NOTIMPLEMENTED();
    return false;
  }

  constexpr bool operator!=(const PrimeFieldFallback& other) const {
    NOTIMPLEMENTED();
    return false;
  }

  constexpr bool operator<(const PrimeFieldFallback& other) const {
    NOTIMPLEMENTED();
    return false;
  }

  constexpr bool operator>(const PrimeFieldFallback& other) const {
    NOTIMPLEMENTED();
    return false;
  }

  constexpr bool operator<=(const PrimeFieldFallback& other) const {
    NOTIMPLEMENTED();
    return false;
  }

  constexpr bool operator>=(const PrimeFieldFallback& other) const {
    NOTIMPLEMENTED();
    return false;
  }

  // AdditiveGroup methods
  constexpr F& NegativeInPlace() {
    NOTIMPLEMENTED();
    return static_cast<F&>(*this);
  }

  // MultiplicativeGroup methods
  F& InverseInPlace() {
    NOTIMPLEMENTED();
    return static_cast<F&>(*this);
  }

  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  mpz_class DivBy2Exp(uint64_t exp) const {
    mpz_class ret;
    NOTIMPLEMENTED();
    return ret;
  }

 private:
  friend class AdditiveMonoid<F>;
  friend class AdditiveGroup<F>;
  friend class MultiplicativeMonoid<F>;
  friend class MultiplicativeGroup<F>;
  friend class PrimeFieldBase<F>;

  // AdditiveMonoid methods
  constexpr F Add(const F& other) const {
    F f;
    NOTIMPLEMENTED();
    return f;
  }

  constexpr F& AddInPlace(const F& other) {
    NOTIMPLEMENTED();
    return static_cast<F&>(*this);
  }

  // AdditiveGroup methods
  constexpr F Sub(const F& other) const {
    F f;
    NOTIMPLEMENTED();
    return f;
  }

  constexpr F& SubInPlace(const F& other) {
    NOTIMPLEMENTED();
    return static_cast<F&>(*this);
  }

  // MultiplicativeMonoid methods
  F Mul(const F& other) const {
    F f;
    NOTIMPLEMENTED();
    return f;
  }

  F& MulInPlace(const F& other) {
    NOTIMPLEMENTED();
    return static_cast<F&>(*this);
  }

  // MultiplicativeGroup methods
  F Div(const F& other) const {
    F f;
    NOTIMPLEMENTED();
    return f;
  }

  F& DivInPlace(const F& other) {
    NOTIMPLEMENTED();
    return static_cast<F&>(*this);
  }

  uint64_t limbs_[LIMB_NUMS] = {
      0,
  };
};

template <typename F, size_t MODULUS_BITS>
std::ostream& operator<<(std::ostream& os,
                         const PrimeFieldFallback<F, MODULUS_BITS>& f) {
  return os << f.ToString();
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_FALLBACK_H_
