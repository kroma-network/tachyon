#ifndef TACHYON_MATH_GEOMETRY_POINT2_H_
#define TACHYON_MATH_GEOMETRY_POINT2_H_

#include "absl/strings/substitute.h"

namespace tachyon::math {

template <typename T>
struct Point2 {
  using value_type = T;

  T x;
  T y;

  constexpr Point2() : Point2(T::Zero(), T::Zero()) {}
  constexpr Point2(const T& x, const T& y) : x(x), y(y) {}
  constexpr Point2(T&& x, T&& y) : x(std::move(x)), y(std::move(y)) {}

  constexpr bool operator==(const Point2& other) const {
    return x == other.x && y == other.y;
  }

  constexpr bool operator!=(const Point2& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", x.ToString(), y.ToString());
  }

  std::string ToHexString() const {
    return absl::Substitute("($0, $1)", x.ToHexString(), y.ToHexString());
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Point2<T>& point) {
  return os << point.ToString();
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_GEOMETRY_POINT2_H_
