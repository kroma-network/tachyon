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

template <typename CharTy>
class Copyable<std::basic_string_view<CharTy>> {
 public:
  static bool WriteTo(const std::basic_string_view<CharTy>& value,
                      Buffer* buffer) {
    if (!buffer->Write(value.size())) return false;
    return buffer->Write(reinterpret_cast<const uint8_t*>(value.data()),
                         value.size());
  }

  static bool ReadFrom(const Buffer& buffer,
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

  static bool ReadFrom(const Buffer& buffer, std::basic_string<CharTy>* value) {
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

  static bool ReadFrom(const Buffer& buffer, const CharTy** value) {
    NOTREACHED() << "Not supported ReadFrom for const CharTy*";
    return false;
  }

  static size_t EstimateSize(const CharTy* value) {
    return sizeof(size_t) +
           std::char_traits<CharTy>::length(value) * sizeof(CharTy);
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

  static bool ReadFrom(const Buffer& buffer, std::vector<T>* values) {
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
                             return total + Copyable<T>::EstimateSize(value);
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

  static bool ReadFrom(const Buffer& buffer, std::array<T, N>* values) {
    for (T& value : (*values)) {
      if (!buffer.Read(&value)) return false;
    }
    return true;
  }

  static size_t EstimateSize(const std::array<T, N>& values) {
    return std::accumulate(values.begin(), values.end(), 0,
                           [](size_t total, const T& value) {
                             return total + Copyable<T>::EstimateSize(value);
                           });
  }
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_COPYABLE_H_
