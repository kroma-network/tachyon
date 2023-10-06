#ifndef TACHYON_MATH_GEOMETRY_POINT4_H_
#define TACHYON_MATH_GEOMETRY_POINT4_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

namespace tachyon::math {

template <typename T>
struct Point4 {
  using value_type = T;

  T x;
  T y;
  T z;
  T w;

  constexpr Point4() : Point4(T::Zero(), T::Zero(), T::Zero(), T::Zero()) {}
  constexpr Point4(const T& x, const T& y, const T& z, const T& w)
      : x(x), y(y), z(z), w(w) {}
  constexpr Point4(T&& x, T&& y, T&& z, T&& w)
      : x(std::move(x)), y(std::move(y)), z(std::move(z)), w(std::move(w)) {}

  constexpr bool operator==(const Point4& other) const {
    return x == other.x && y == other.y && z == other.z && w == other.w;
  }

  constexpr bool operator!=(const Point4& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2, $3)", x.ToString(), y.ToString(),
                            z.ToString(), w.ToString());
  }

  std::string ToHexString() const {
    return absl::Substitute("($0, $1, $2, $3)", x.ToHexString(),
                            y.ToHexString(), z.ToHexString(), w.ToHexString());
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Point4<T>& point) {
  return os << point.ToString();
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_GEOMETRY_POINT4_H_
