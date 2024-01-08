#ifndef TACHYON_MATH_BASE_GMP_BIT_TRAITS_H_
#define TACHYON_MATH_BASE_GMP_BIT_TRAITS_H_

#include "tachyon/math/base/bit_traits_forward.h"
#include "tachyon/math/base/gmp/gmp_util.h"

namespace tachyon::math {

template <>
class BitTraits<mpz_class> {
 public:
  constexpr static bool kIsDynamic = true;

  static size_t GetNumBits(const mpz_class& value) {
    return gmp::GetNumBits(value);
  }

  static bool TestBit(const mpz_class& value, size_t index) {
    return gmp::TestBit(value, index);
  }

  static void SetBit(mpz_class& value, size_t index, bool bit_value) {
    return gmp::SetBit(value, index, bit_value);
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_GMP_BIT_TRAITS_H_
