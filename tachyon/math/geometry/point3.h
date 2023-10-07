#ifndef TACHYON_MATH_GEOMETRY_POINT3_H_
#define TACHYON_MATH_GEOMETRY_POINT3_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

namespace tachyon::math {

// |Point3| represents points expanded into 3-element vectors.
// It can represent elliptic curve points in either Projective or Jacobian form.
template <typename T>
struct Point3 {
  using value_type = T;

  T x;
  T y;
  T z;

  constexpr Point3() : Point3(T::Zero(), T::Zero(), T::Zero()) {}
  constexpr Point3(const T& x, const T& y, const T& z) : x(x), y(y), z(z) {}
  constexpr Point3(T&& x, T&& y, T&& z)
      : x(std::move(x)), y(std::move(y)), z(std::move(z)) {}

  constexpr bool operator==(const Point3& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  constexpr bool operator!=(const Point3& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2)", x.ToString(), y.ToString(),
                            z.ToString());
  }

  std::string ToHexString() const {
    return absl::Substitute("($0, $1, $2)", x.ToHexString(), y.ToHexString(),
                            z.ToHexString());
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Point3<T>& point) {
  return os << point.ToString();
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_GEOMETRY_POINT3_H_
