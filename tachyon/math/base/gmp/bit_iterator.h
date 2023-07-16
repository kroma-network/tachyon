#ifndef TACHYON_MATH_BASE_GMP_BIT_TRAITS_H_
#define TACHYON_MATH_BASE_GMP_BIT_TRAITS_H_

#include "tachyon/math/base/bit_iterator.h"
#include "tachyon/math/base/gmp/gmp_util.h"

namespace tachyon {
namespace math {

template <>
class BitTraits<mpz_class> {
 public:
  static size_t GetNumBits(const mpz_class& value) {
    return gmp::GetNumBits(value);
  }

  static bool TestBit(const mpz_class& value, size_t index) {
    return gmp::TestBit(value, index);
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_GMP_BIT_TRAITS_H_
