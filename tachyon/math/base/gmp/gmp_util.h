#ifndef TACHYON_MATH_BASE_GMP_GMP_UTIL_H_
#define TACHYON_MATH_BASE_GMP_GMP_UTIL_H_

#include <stddef.h>

#include <string_view>
#include <type_traits>

#include "third_party/gmp/include/gmpxx.h"

#include "tachyon/math/base/sign.h"

namespace tachyon::math::gmp {

// Generate a uniform integer in the range 0 to n-1, inclusive.
TACHYON_EXPORT mpz_class Random(mpz_class n);

TACHYON_EXPORT bool ParseIntoMpz(std::string_view str, int base,
                                 mpz_class* out);

TACHYON_EXPORT void MustParseIntoMpz(std::string_view str, int base,
                                     mpz_class* out);

TACHYON_EXPORT mpz_class FromDecString(std::string_view str);
TACHYON_EXPORT mpz_class FromHexString(std::string_view str);

template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
mpz_class FromUnsignedInt(T value) {
  mpz_class ret;
  mpz_set_ui(ret.get_mpz_t(), value);
  return ret;
}

template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
mpz_class FromSignedInt(T value) {
  mpz_class ret;
  mpz_set_si(ret.get_mpz_t(), value);
  return ret;
}

TACHYON_EXPORT Sign GetSign(const mpz_class& value);
TACHYON_EXPORT bool IsZero(const mpz_class& value);
TACHYON_EXPORT bool IsNegative(const mpz_class& value);
TACHYON_EXPORT bool IsPositive(const mpz_class& value);

TACHYON_EXPORT mpz_class GetAbs(const mpz_class& value);

TACHYON_EXPORT size_t GetNumBits(const mpz_class& value);
TACHYON_EXPORT bool TestBit(const mpz_class& value, size_t index);
// This calls either SetBit(value, index) or ClearBit(value, index)
// based on the value of bit_value.
TACHYON_EXPORT void SetBit(mpz_class& value, size_t index, bool bit_value);

TACHYON_EXPORT void SetBit(mpz_class& value, size_t index);
TACHYON_EXPORT void ClearBit(mpz_class& value, size_t index);

TACHYON_EXPORT uint64_t* GetLimbs(const mpz_class& value);
TACHYON_EXPORT size_t GetLimbSize(const mpz_class& value);
TACHYON_EXPORT const mp_limb_t& GetLimbConstRef(const mpz_class& value,
                                                size_t idx);
TACHYON_EXPORT mp_limb_t& GetLimbRef(mpz_class& value, size_t idx);
TACHYON_EXPORT void CopyLimbs(const mpz_class& value, uint64_t* limbs);
TACHYON_EXPORT void WriteLimbs(const uint64_t* limbs, size_t limb_size,
                               mpz_class* out);

TACHYON_EXPORT mpz_class DivBy2Exp(const mpz_class& value, uint64_t exp);

}  // namespace tachyon::math::gmp

#endif  // TACHYON_MATH_BASE_GMP_GMP_UTIL_H_
