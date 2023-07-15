#include "tachyon/math/base/sign.h"

#include "tachyon/base/strings/string_util.h"

namespace tachyon {
namespace math {

std::string SignToString(Sign sign) {
  switch (sign) {
    case Sign::kZero:
      return "0";
    case Sign::kPositive:
      return "+";
    case Sign::kNegative:
      return "-";
    case Sign::kNaN:
      return "nan";
  }
  NOTREACHED();
  return base::EmptyString();
}

}  // namespace math
}  // namespace tachyon