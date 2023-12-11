#ifndef TACHYON_MATH_GEOMETRY_POINT4_H_
#define TACHYON_MATH_GEOMETRY_POINT4_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "tachyon/base/buffer/copyable.h"

namespace tachyon {
namespace math {

// |Point4| represents points expanded into 4-element vectors.
// |Point4| is used to represent elliptic curve points in the XYZZ form.
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

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1, $2, $3)", x.ToHexString(pad_zero),
                            y.ToHexString(pad_zero), z.ToHexString(pad_zero),
                            w.ToHexString(pad_zero));
  }
};

}  // namespace math

namespace base {

template <typename T>
class Copyable<math::Point4<T>> {
 public:
  static bool WriteTo(const math::Point4<T>& point, Buffer* buffer) {
    return buffer->WriteMany(point.x, point.y, point.z, point.w);
  }

  static bool ReadFrom(const Buffer& buffer, math::Point4<T>* point) {
    return buffer.ReadMany(&point->x, &point->y, &point->z, &point->w);
  }

  static size_t EstimateSize(const math::Point4<T>& point) {
    return base::EstimateSize(point.x) + base::EstimateSize(point.y) +
           base::EstimateSize(point.z) + base::EstimateSize(point.w);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_GEOMETRY_POINT4_H_
