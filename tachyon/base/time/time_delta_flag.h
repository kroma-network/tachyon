#ifndef TACHYON_BASE_TIME_TIME_DELTA_FLAG_H_
#define TACHYON_BASE_TIME_TIME_DELTA_FLAG_H_

#include <functional>
#include <string>

#include "tachyon/base/flag/flag.h"
#include "tachyon/base/numerics/safe_conversions.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/base/time/time.h"

namespace tachyon::base {

template <>
class FlagValueTraits<TimeDelta> {
 public:
  static bool ParseValue(std::string_view input, TimeDelta* value,
                         std::string* reason) {
    std::function<TimeDelta(int)> func_from_int;
    std::function<TimeDelta(int64_t)> func_from_int64;
    std::function<TimeDelta(double)> func_from_double;
    if (ConsumeSuffix(&input, "d")) {
      func_from_int = &Days<int>;
    } else if (ConsumeSuffix(&input, "h")) {
      func_from_int = &Hours<int>;
    } else if (ConsumeSuffix(&input, "m")) {
      func_from_int = &Minutes<int>;
    } else if (ConsumeSuffix(&input, "ms")) {
      func_from_int64 = &Milliseconds<int64_t>;
      func_from_double = &Milliseconds<double>;
    } else if (ConsumeSuffix(&input, "us")) {
      func_from_int64 = &Microseconds<int64_t>;
      func_from_double = &Microseconds<double>;
    } else if (ConsumeSuffix(&input, "ns")) {
      func_from_int64 = &Nanoseconds<int64_t>;
      func_from_double = &Nanoseconds<double>;
    } else if (ConsumeSuffix(&input, "s")) {
      func_from_int64 = &Seconds<int64_t>;
      func_from_double = &Seconds<double>;
    } else {
      *reason = "no suffix!, please add suffix d, h, m, s, ms, us or ns";
      return false;
    }

    int64_t int64_value;
    bool success_to_convert_int64 = StringToInt64(input, &int64_value);
    if (func_from_int) {
      if (success_to_convert_int64 &&
          IsValueInRangeForNumericType<int>(int64_value)) {
        *value = func_from_int(static_cast<int>(int64_value));
        return true;
      }
      *reason = "failed to convert to int";
      return false;
    }
    if (success_to_convert_int64) {
      *value = func_from_int64(int64_value);
      return true;
    }
    double double_value;
    if (!StringToDouble(input, &double_value)) {
      *reason = "failed to convert to double";
      return false;
    }
    *value = func_from_double(double_value);
    return true;
  }
};

typedef Flag<TimeDelta> TimeDeltaFlag;

}  // namespace tachyon::base

#endif  // TACHYON_BASE_TIME_TIME_DELTA_FLAG_H_
