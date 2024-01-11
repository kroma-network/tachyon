#ifndef TACHYON_MATH_GEOMETRY_POINT2_H_
#define TACHYON_MATH_GEOMETRY_POINT2_H_

#include <string>
#include <utility>

#include "absl/strings/substitute.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/json/json.h"

namespace tachyon {
namespace math {

// |Point2| represents points expanded into 2-element vectors.
// It can represent elliptic curve points in Affine form.
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

  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1)", x.ToHexString(pad_zero),
                            y.ToHexString(pad_zero));
  }
};

}  // namespace math

namespace base {

template <typename T>
class Copyable<math::Point2<T>> {
 public:
  static bool WriteTo(const math::Point2<T>& point, Buffer* buffer) {
    return buffer->WriteMany(point.x, point.y);
  }

  static bool ReadFrom(const Buffer& buffer, math::Point2<T>* point) {
    return buffer.ReadMany(&point->x, &point->y);
  }

  static size_t EstimateSize(const math::Point2<T>& point) {
    return base::EstimateSize(point.x) + base::EstimateSize(point.y);
  }
};

template <typename T>
class RapidJsonValueConverter<math::Point2<T>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const math::Point2<T>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "x", value.x, allocator);
    AddJsonElement(object, "y", value.y, allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::Point2<T>* value, std::string* error) {
    T x;
    T y;
    if (!ParseJsonElement(json_value, "x", &x, error)) return false;
    if (!ParseJsonElement(json_value, "y", &y, error)) return false;
    value->x = std::move(x);
    value->y = std::move(y);
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_GEOMETRY_POINT2_H_
