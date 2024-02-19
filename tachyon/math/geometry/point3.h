#ifndef TACHYON_MATH_GEOMETRY_POINT3_H_
#define TACHYON_MATH_GEOMETRY_POINT3_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/json/json.h"

namespace tachyon {
namespace math {

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

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1, $2)", x.ToHexString(pad_zero),
                            y.ToHexString(pad_zero), z.ToHexString(pad_zero));
  }
};

}  // namespace math

namespace base {

template <typename T>
class Copyable<math::Point3<T>> {
 public:
  static bool WriteTo(const math::Point3<T>& point, Buffer* buffer) {
    return buffer->WriteMany(point.x, point.y, point.z);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, math::Point3<T>* point) {
    return buffer.ReadMany(&point->x, &point->y, &point->z);
  }

  static size_t EstimateSize(const math::Point3<T>& point) {
    return base::EstimateSize(point.x, point.y, point.z);
  }
};

template <typename T>
class RapidJsonValueConverter<math::Point3<T>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const math::Point3<T>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "x", value.x, allocator);
    AddJsonElement(object, "y", value.y, allocator);
    AddJsonElement(object, "z", value.z, allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::Point3<T>* value, std::string* error) {
    T x;
    T y;
    T z;
    if (!ParseJsonElement(json_value, "x", &x, error)) return false;
    if (!ParseJsonElement(json_value, "y", &y, error)) return false;
    if (!ParseJsonElement(json_value, "z", &z, error)) return false;
    value->x = std::move(x);
    value->y = std::move(y);
    value->z = std::move(z);
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_GEOMETRY_POINT3_H_
