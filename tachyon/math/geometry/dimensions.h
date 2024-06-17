#ifndef TACHYON_MATH_GEOMETRY_DIMENSIONS_H_
#define TACHYON_MATH_GEOMETRY_DIMENSIONS_H_

#include <stddef.h>

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/json/json.h"
#include "tachyon/export.h"

namespace tachyon {
namespace math {

// NOTE(chokobole): |Dimensions| class is copyable, assignable, and occupies 128
// bits per instance. Prefer to pass them by value.
struct TACHYON_EXPORT Dimensions {
  size_t width = 0;
  size_t height = 0;

  constexpr Dimensions() = default;
  constexpr Dimensions(size_t width, size_t height)
      : width(width), height(height) {}

  constexpr bool operator==(Dimensions other) const {
    return width == other.width && height == other.height;
  }

  constexpr bool operator!=(Dimensions other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", width, height);
  }
};

}  // namespace math

namespace base {

template <>
class Copyable<math::Dimensions> {
 public:
  static bool WriteTo(math::Dimensions dimensions, Buffer* buffer) {
    return buffer->WriteMany(dimensions.width, dimensions.height);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       math::Dimensions* dimensions) {
    return buffer.ReadMany(&dimensions->width, &dimensions->height);
  }

  static size_t EstimateSize(math::Dimensions dimensions) {
    return base::EstimateSize(dimensions.width, dimensions.height);
  }
};

template <>
class RapidJsonValueConverter<math::Dimensions> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(math::Dimensions value, Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "width", value.width, allocator);
    AddJsonElement(object, "height", value.height, allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 math::Dimensions* value, std::string* error) {
    size_t width;
    size_t height;
    if (!ParseJsonElement(json_value, "width", &width, error)) return false;
    if (!ParseJsonElement(json_value, "height", &height, error)) return false;
    value->width = width;
    value->height = height;
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_GEOMETRY_DIMENSIONS_H_
