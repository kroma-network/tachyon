#ifndef TACHYON_BASE_BUFFER_COPYABLE_H_
#define TACHYON_BASE_BUFFER_COPYABLE_H_

#include <array>
#include <numeric>
#include <string>
#include <vector>

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/base/buffer/copyable_forward.h"
#include "tachyon/base/logging.h"

namespace tachyon::base {

// NOTE(chokobole): We omitted |WriteTo()| and |ReadFrom()| on purpose.
// The reason why |EstimateSize()| is implemented here is to enable other
// |Copyable<T>| to use this. (e.g, Copyable<int>::EstimateSize(1))
template <typename T>
class Copyable<T, std::enable_if_t<internal::IsBuiltinSerializable<T>::value>> {
 public:
  static size_t EstimateSize(const T& value) { return sizeof(T); }
};

template <typename T>
class Copyable<T, std::enable_if_t<std::is_enum_v<T>>> {
 public:
  static bool WriteTo(const T& value, Buffer* buffer) {
    return buffer->Write(static_cast<std::underlying_type_t<T>>(value));
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, T* value) {
    std::underlying_type_t<T> underlying_value;
    if (!buffer.Read(&underlying_value)) return false;
    *value = static_cast<T>(underlying_value);
    return true;
  }

  static size_t EstimateSize(T value) { return sizeof(T); }
};

template <typename CharTy>
class Copyable<std::basic_string_view<CharTy>> {
 public:
  static bool WriteTo(const std::basic_string_view<CharTy>& value,
                      Buffer* buffer) {
    if (!buffer->Write(value.size())) return false;
    return buffer->Write(reinterpret_cast<const uint8_t*>(value.data()),
                         value.size());
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       std::basic_string_view<CharTy>* value) {
    NOTREACHED() << "Not supported ReadFrom for std::basic_string_view<CharTy>";
    return false;
  }

  static size_t EstimateSize(const std::basic_string_view<CharTy>& value) {
    return sizeof(size_t) + value.size();
  }
};

template <typename CharTy>
class Copyable<std::basic_string<CharTy>> {
 public:
  static bool WriteTo(const std::basic_string<CharTy>& value, Buffer* buffer) {
    if (!buffer->Write(value.size())) return false;
    return buffer->Write(reinterpret_cast<const uint8_t*>(value.data()),
                         value.size());
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       std::basic_string<CharTy>* value) {
    size_t size;
    if (!buffer.Read(&size)) return false;
    value->resize(size);
    return buffer.Read(reinterpret_cast<uint8_t*>(value->data()), size);
  }

  static size_t EstimateSize(const std::basic_string<CharTy>& value) {
    return sizeof(size_t) + value.size();
  }
};

template <typename CharTy>
class Copyable<
    const CharTy*,
    std::enable_if_t<
        std::is_same_v<CharTy, char> || std::is_same_v<CharTy, wchar_t> ||
        std::is_same_v<CharTy, char16_t> || std::is_same_v<CharTy, char32_t>>> {
 public:
  static bool WriteTo(const CharTy* value, Buffer* buffer) {
    size_t length = std::char_traits<CharTy>::length(value);
    if (!buffer->Write(length)) return false;
    return buffer->Write(reinterpret_cast<const uint8_t*>(value),
                         length * sizeof(CharTy));
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, const CharTy** value) {
    NOTREACHED() << "Not supported ReadFrom for const CharTy*";
    return false;
  }

  static size_t EstimateSize(const CharTy* value) {
    return sizeof(size_t) +
           std::char_traits<CharTy>::length(value) * sizeof(CharTy);
  }
};

template <typename T, size_t N>
class Copyable<T[N]> {
 public:
  static bool WriteTo(const T* values, Buffer* buffer) {
    for (size_t i = 0; i < N; ++i) {
      if (!buffer->Write(values[i])) return false;
    }
    return true;
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, T* values) {
    for (size_t i = 0; i < N; ++i) {
      if (!buffer.Read(&values[i])) return false;
    }
    return true;
  }

  static size_t EstimateSize(const T* values) {
    return std::accumulate(values, &values[N], 0,
                           [](size_t total, const T& value) {
                             return total + base::EstimateSize(value);
                           });
  }
};

template <typename T>
class Copyable<std::vector<T>> {
 public:
  static bool WriteTo(const std::vector<T>& values, Buffer* buffer) {
    if (!buffer->Write(values.size())) return false;
    for (const T& value : values) {
      if (!buffer->Write(value)) return false;
    }
    return true;
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, std::vector<T>* values) {
    size_t size;
    if (!buffer.Read(&size)) return false;
    values->resize(size);
    for (T& value : (*values)) {
      if (!buffer.Read(&value)) return false;
    }
    return true;
  }

  static size_t EstimateSize(const std::vector<T>& values) {
    return std::accumulate(values.begin(), values.end(), sizeof(size_t),
                           [](size_t total, const T& value) {
                             return total + base::EstimateSize(value);
                           });
  }
};

template <typename T, size_t N>
class Copyable<std::array<T, N>> {
 public:
  static bool WriteTo(const std::array<T, N>& values, Buffer* buffer) {
    for (const T& value : values) {
      if (!buffer->Write(value)) return false;
    }
    return true;
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, std::array<T, N>* values) {
    for (T& value : (*values)) {
      if (!buffer.Read(&value)) return false;
    }
    return true;
  }

  static size_t EstimateSize(const std::array<T, N>& values) {
    return std::accumulate(values.begin(), values.end(), 0,
                           [](size_t total, const T& value) {
                             return total + base::EstimateSize(value);
                           });
  }
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_COPYABLE_H_
