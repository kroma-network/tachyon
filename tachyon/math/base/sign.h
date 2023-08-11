#ifndef TACHYON_MATH_BASE_SIGN_H_
#define TACHYON_MATH_BASE_SIGN_H_

#include <cmath>
#include <string>
#include <type_traits>

#include "tachyon/base/logging.h"

namespace tachyon::math {

enum class Sign {
  kZero,
  kPositive,
  kNegative,
  // Not a number. GetSign(nan) gives this.
  kNaN,
};

TACHYON_EXPORT std::string SignToString(Sign sign);

//  0: kZero
//  1: kPositive
// -1: kNegative
constexpr Sign ToSign(int v) {
  switch (v) {
    case 0:
      return Sign::kZero;
    case 1:
      return Sign::kPositive;
    case -1:
      return Sign::kNegative;
  }
  NOTREACHED();
  return Sign::kNaN;
}

// Taken from
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T>
constexpr Sign GetSign(T x) {
  if constexpr (std::is_signed_v<T>) {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isnan(x)) return Sign::kNaN;
    }
    return ToSign((T(0) < x) - (x < T(0)));
  } else {
    return ToSign(T(0) < x);
  }
}

template <typename T, typename SFINAE = void>
class SignedValue;

}  // namespace tachyon::math

#endif  // TACHYON_MATH_BASE_SIGN_H_
