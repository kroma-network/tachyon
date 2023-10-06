#ifndef TACHYON_MATH_BASE_GMP_SIGNED_VALUE_H_
#define TACHYON_MATH_BASE_GMP_SIGNED_VALUE_H_

#include "tachyon/math/base/gmp/gmp_util.h"
#include "tachyon/math/base/sign.h"

namespace tachyon::math {

template <>
class SignedValue<mpz_class> {
 public:
  Sign sign;
  mpz_class abs_value;

  SignedValue() = default;
  explicit SignedValue(const mpz_class& value)
      : sign(gmp::GetSign(value)), abs_value(gmp::GetAbs(value)) {}

  mpz_class ToValue() const {
    switch (sign) {
      case Sign::kZero:
      case Sign::kPositive:
        return abs_value;
      case Sign::kNegative:
        return -abs_value;
      case Sign::kNaN:
        NOTREACHED();
        return abs_value;
    }
    NOTREACHED();
    return abs_value;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_GMP_SIGNED_VALUE_H_
