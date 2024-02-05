#ifndef TACHYON_MATH_GEOMETRY_POINT4_H_
#define TACHYON_MATH_GEOMETRY_POINT4_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/json/json.h"

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

  static bool ReadFrom(const ReadOnlyBuffer& buffer, math::Point4<T>* point) {
    return buffer.ReadMany(&point->x, &point->y, &point->z, &point->w);
  }

  static size_t EstimateSize(const math::Point4<T>& point) {
    return base::EstimateSize(point.x) + base::EstimateSize(point.y) +
           base::EstimateSize(point.z) + base::EstimateSize(point.w);
  }
};

template <typename T>
class RapidJsonValueConverter<math::Point4<T>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const math::Point4<T>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "x", value.x, allocator);
    AddJsonElement(object, "y", value.y, allocator);
    AddJsonElement(object, "z", value.z, allocator);
    AddJsonElement(object, "w", value.w, allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::Point4<T>* value, std::string* error) {
    T x;
    T y;
    T z;
    T w;
    if (!ParseJsonElement(json_value, "x", &x, error)) return false;
    if (!ParseJsonElement(json_value, "y", &y, error)) return false;
    if (!ParseJsonElement(json_value, "z", &z, error)) return false;
    if (!ParseJsonElement(json_value, "w", &w, error)) return false;
    value->x = std::move(x);
    value->y = std::move(y);
    value->z = std::move(z);
    value->w = std::move(w);
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_GEOMETRY_POINT4_H_
