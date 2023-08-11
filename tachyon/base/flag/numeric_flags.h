#ifndef TACHYON_BASE_FLAG_NUMERIC_FLAGS_H_
#define TACHYON_BASE_FLAG_NUMERIC_FLAGS_H_

#include "tachyon/base/flag/flag.h"

namespace tachyon::base {
namespace flag_internal {

template <typename T>
struct PositiveNumber {
  T value;
};

using PositiveInt = flag_internal::PositiveNumber<int>;
using PositiveInt8 = flag_internal::PositiveNumber<int8_t>;
using PositiveInt16 = flag_internal::PositiveNumber<int16_t>;
using PositiveInt32 = flag_internal::PositiveNumber<int32_t>;
using PositiveInt64 = flag_internal::PositiveNumber<int64_t>;
using PositiveFloat = flag_internal::PositiveNumber<float>;
using PositiveDouble = flag_internal::PositiveNumber<double>;

}  // namespace flag_internal

template <typename T>
class FlagValueTraits<flag_internal::PositiveNumber<T>> {
 public:
  static bool ParseValue(std::string_view input,
                         flag_internal::PositiveNumber<T>* value,
                         std::string* reason) {
    T n;
    if (!FlagValueTraits<T>::ParseValue(input, &n, reason)) {
      return false;
    }
    if (n > 0) {
      value->value = n;
      return true;
    } else {
      *reason = "value should be positive";
      return false;
    }
    return true;
  }
};

using PositiveIntFlag = Flag<flag_internal::PositiveInt>;
using PositiveInt8Flag = Flag<flag_internal::PositiveInt8>;
using PositiveInt16Flag = Flag<flag_internal::PositiveInt16>;
using PositiveInt32Flag = Flag<flag_internal::PositiveInt32>;
using PositiveInt64Flag = Flag<flag_internal::PositiveInt64>;
using PositiveFloatFlag = Flag<flag_internal::PositiveFloat>;
using PositiveDoubleFlag = Flag<flag_internal::PositiveDouble>;

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FLAG_NUMERIC_FLAGS_H_
