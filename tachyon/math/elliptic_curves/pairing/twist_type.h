#ifndef TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_TWIST_TYPE_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_TWIST_TYPE_H_

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/base/flag/flag_value_traits.h"
#include "tachyon/base/logging.h"

namespace tachyon {
namespace math {

enum class TwistType {
  kM,
  kD,
};

inline const char* TwistTypeToString(TwistType type) {
  switch (type) {
    case TwistType::kM:
      return "M";
    case TwistType::kD:
      return "D";
  }
  NOTREACHED();
  return "";
}

}  // namespace math

namespace base {

template <>
class FlagValueTraits<math::TwistType> {
 public:
  static bool ParseValue(std::string_view input, math::TwistType* value,
                         std::string* reason) {
    if (input == "M") {
      *value = math::TwistType::kM;
    } else if (input == "D") {
      *value = math::TwistType::kD;
    } else {
      *reason = absl::Substitute("Unknown twist type: $0", input);
      return false;
    }
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_PAIRING_TWIST_TYPE_H_
