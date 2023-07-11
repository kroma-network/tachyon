#ifndef TACHYON_MATH_BASE_DIVISION_RESULT_H_
#define TACHYON_MATH_BASE_DIVISION_RESULT_H_

#include <ostream>

#include "absl/strings/substitute.h"

namespace tachyon {
namespace math {

template <typename T>
struct DivisionResult {
  T quotient;
  T remainder;

  bool operator==(const DivisionResult& other) const {
    return quotient == other.quotient && remainder == other.remainder;
  }
  bool operator!=(const DivisionResult& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", quotient.ToString(),
                            remainder.ToString());
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const DivisionResult<T>& result) {
  return os << result.ToString();
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_DIVISION_RESULT_H_
