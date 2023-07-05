#ifndef TACHYON_MATH_GEOMETRY_POINT2_H_
#define TACHYON_MATH_GEOMETRY_POINT2_H_

#include "absl/strings/substitute.h"

namespace tachyon {
namespace math {

template <typename T>
struct Point2 {
  using value_type = T;

  T x;
  T y;

  constexpr Point2() : Point2(0, 0) {}
  constexpr Point2(T x, T y) : x(x), y(y) {}

  constexpr bool operator==(const Point2& other) const {
    return x == other.x && y == other.y;
  }

  constexpr bool operator!=(const Point2& other) const {
    return !operator==(other);
  }

  std::string ToString() const { return absl::Substitute("($0, $1)", x, y); }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Point2<T>& point) {
  return os << point.ToString();
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_GEOMETRY_POINT2_H_