#ifndef TACHYON_MATH_BASE_GMP_CONVERSIONS_H_
#define TACHYON_MATH_BASE_GMP_CONVERSIONS_H_

#include "tachyon/math/base/conversions.h"
#include "tachyon/math/base/gmp/gmp_util.h"

namespace tachyon::math {

template <>
class StringNumberConversion<mpz_class> {
 public:
  static mpz_class FromDecString(std::string_view str) {
    return gmp::FromDecString(str);
  }

  static mpz_class FromHexString(std::string_view str) {
    return gmp::FromHexString(str);
  }
};

template <>
class IntConversion<mpz_class> {
 public:
  template <typename T>
  static mpz_class FromInt(T v) {
    if (constexpr std::is_signed_v<T>) {
      return gmp::FromSignedInt(v);
    } else {
      return gmp::FromUnsignedInt(v);
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_GMP_CONVERSIONS_H_
