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

template <typename _Config>
class PrimeFieldFallback : public PrimeFieldBase<PrimeFieldFallback<_Config>> {
 public:
  static constexpr size_t kModulusBits = _Config::kModulusBits;
  static constexpr size_t kLimbNums = (kModulusBits + 63) / 64;

  using Config = _Config;
  using value_type = uint64_t[kLimbNums];

  constexpr PrimeFieldFallback() = default;
  constexpr explicit PrimeFieldFallback(uint64_t value) { limbs_[0] = value; }
  constexpr explicit PrimeFieldFallback(uint64_t limbs[kLimbNums])
      : limbs_(limbs) {}
  constexpr PrimeFieldFallback(const PrimeFieldFallback& other) = default;
  constexpr PrimeFieldFallback& operator=(const PrimeFieldFallback& other) =
      default;
  constexpr PrimeFieldFallback(PrimeFieldFallback&& other) = default;
  constexpr PrimeFieldFallback& operator=(PrimeFieldFallback&& other) = default;

  constexpr static PrimeFieldFallback Zero() { return PrimeFieldFallback(); }

  constexpr static PrimeFieldFallback One() { return PrimeFieldFallback(1); }

  constexpr static PrimeFieldFallback FromDecString(std::string_view str) {
    NOTIMPLEMENTED();
    return PrimeFieldFallback();
  }
  constexpr static PrimeFieldFallback FromHexString(std::string_view str) {
    NOTIMPLEMENTED();
    return PrimeFieldFallback();
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

  // This is needed by MSM.
  // See tachyon/math/elliptic_curves/msm/variable_base_msm.h
  mpz_class DivBy2Exp(uint64_t exp) const {
    mpz_class ret;
    NOTIMPLEMENTED();
    return ret;
  }

  // AdditiveMonoid methods
  constexpr PrimeFieldFallback Add(const PrimeFieldFallback& other) const {
    PrimeFieldFallback f;
    NOTIMPLEMENTED();
    return f;
  }

  constexpr PrimeFieldFallback& AddInPlace(const PrimeFieldFallback& other) {
    NOTIMPLEMENTED();
    return *this;
  }

  PrimeFieldFallback& DoubleInPlace() {
    NOTIMPLEMENTED();
    return *this;
  }

  // AdditiveGroup methods
  constexpr PrimeFieldFallback Sub(const PrimeFieldFallback& other) const {
    PrimeFieldFallback f;
    NOTIMPLEMENTED();
    return f;
  }

  constexpr PrimeFieldFallback& SubInPlace(const PrimeFieldFallback& other) {
    NOTIMPLEMENTED();
    return *this;
  }

  PrimeFieldFallback& NegInPlace() {
    NOTIMPLEMENTED();
    return *this;
  }

  // MultiplicativeMonoid methods
  PrimeFieldFallback Mul(const PrimeFieldFallback& other) const {
    PrimeFieldFallback f;
    NOTIMPLEMENTED();
    return f;
  }

  PrimeFieldFallback& MulInPlace(const PrimeFieldFallback& other) {
    NOTIMPLEMENTED();
    return *this;
  }

  PrimeFieldFallback& SquareInPlace() { return MulInPlace(*this); }

  // MultiplicativeGroup methods
  PrimeFieldFallback Div(const PrimeFieldFallback& other) const {
    PrimeFieldFallback f;
    NOTIMPLEMENTED();
    return f;
  }

  PrimeFieldFallback& DivInPlace(const PrimeFieldFallback& other) {
    NOTIMPLEMENTED();
    return *this;
  }

  PrimeFieldFallback& InverseInPlace() {
    NOTIMPLEMENTED();
    return *this;
  }

  uint64_t limbs_[kLimbNums] = {
      0,
  };
};

template <typename Config>
std::ostream& operator<<(std::ostream& os,
                         const PrimeFieldFallback<Config>& f) {
  return os << f.ToString();
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_PRIME_FIELD_FALLBACK_H_
