#ifndef TACHYON_BASE_JSON_RAPIDJSON_UTIL_H_
#define TACHYON_BASE_JSON_RAPIDJSON_UTIL_H_

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"

#include "tachyon/base/bit_cast.h"
#include "tachyon/base/numerics/safe_conversions.h"
#include "tachyon/export.h"

namespace tachyon::base {

TACHYON_EXPORT std::string RapidJsonMismatchedTypeError(
    std::string_view key, std::string_view type, const rapidjson::Value& value);

template <typename T>
std::string RapidJsonOutOfRangeError(std::string_view key, T value) {
  return absl::Substitute("value($0) of \"$1\" is out of range", value, key);
}

template <typename T, typename SFINAE = void>
class RapidJsonValueConverter;

template <>
class RapidJsonValueConverter<bool> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(bool value, Allocator& allocator) {
    return rapidjson::Value(value);
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 bool* value, std::string* error) {
    if (!json_value.IsBool()) {
      *error = RapidJsonMismatchedTypeError(key, "bool", json_value);
      return false;
    }
    *value = json_value.GetBool();
    return true;
  }
};

template <typename T>
class RapidJsonValueConverter<
    T, std::enable_if_t<std::is_integral<T>::value &&
                        std::is_signed<T>::value && sizeof(T) == 8>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(T value, Allocator& allocator) {
    int64_t i64_value = base::bit_cast<int64_t>(value);
    return rapidjson::Value(i64_value);
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 T* value, std::string* error) {
    if (!json_value.IsInt64() && !json_value.IsInt()) {
      *error = RapidJsonMismatchedTypeError(key, "int64", json_value);
      return false;
    }
    if (json_value.IsInt()) {
      *value = base::bit_cast<T>(static_cast<int64_t>(json_value.GetInt()));
    } else {
      *value = base::bit_cast<T>(json_value.GetInt64());
    }
    return true;
  }
};

template <typename T>
class RapidJsonValueConverter<
    T, std::enable_if_t<std::is_integral<T>::value &&
                        !std::is_signed<T>::value && sizeof(T) == 8>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(T value, Allocator& allocator) {
    uint64_t u64_value = base::bit_cast<uint64_t>(value);
    return rapidjson::Value(u64_value);
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 T* value, std::string* error) {
    if (!json_value.IsUint64() && !json_value.IsUint()) {
      *error = RapidJsonMismatchedTypeError(key, "uint64", json_value);
      return false;
    }
    if (json_value.IsUint()) {
      *value = base::bit_cast<T>(static_cast<uint64_t>(json_value.GetUint()));
    } else {
      *value = base::bit_cast<T>(json_value.GetUint64());
    }
    return true;
  }
};

template <typename T>
class RapidJsonValueConverter<
    T, std::enable_if_t<std::is_integral<T>::value &&
                        std::is_signed<T>::value && sizeof(T) < 8>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(T value, Allocator& allocator) {
    static_assert(sizeof(T) <= sizeof(int));
    return rapidjson::Value(static_cast<int>(value));
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 T* value, std::string* error) {
    if (!json_value.IsInt()) {
      *error = RapidJsonMismatchedTypeError(key, "int", json_value);
      return false;
    }
    int value_tmp = json_value.GetInt();
    if (!IsValueInRangeForNumericType<T>(value_tmp)) {
      *error = RapidJsonOutOfRangeError(key, value_tmp);
      return false;
    }
    *value = static_cast<T>(value_tmp);
    return true;
  }
};

template <typename T>
class RapidJsonValueConverter<
    T, std::enable_if_t<std::is_integral<T>::value &&
                        !std::is_signed<T>::value && sizeof(T) < 8>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(T value, Allocator& allocator) {
    static_assert(sizeof(T) <= sizeof(unsigned));
    return rapidjson::Value(static_cast<unsigned>(value));
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 T* value, std::string* error) {
    if (!json_value.IsUint()) {
      *error = RapidJsonMismatchedTypeError(key, "uint", json_value);
      return false;
    }
    unsigned value_tmp = json_value.GetUint();
    if (!IsValueInRangeForNumericType<T>(value_tmp)) {
      *error = RapidJsonOutOfRangeError(key, value_tmp);
      return false;
    }
    *value = static_cast<T>(value_tmp);
    return true;
  }
};

template <typename T>
class RapidJsonValueConverter<
    T, std::enable_if_t<std::is_floating_point<T>::value>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(T value, Allocator& allocator) {
    return rapidjson::Value(value);
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 T* value, std::string* error) {
    if (!json_value.IsDouble()) {
      *error = RapidJsonMismatchedTypeError(key, "double", json_value);
      return false;
    }
    double value_tmp = json_value.GetDouble();
    if (!IsValueInRangeForNumericType<T>(value_tmp)) {
      *error = RapidJsonOutOfRangeError(key, value_tmp);
      return false;
    }
    *value = static_cast<T>(value_tmp);
    return true;
  }
};

template <>
class RapidJsonValueConverter<std::string> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const std::string& value, Allocator& allocator) {
    return rapidjson::Value(value.c_str(), value.length(), allocator);
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 std::string* value, std::string* error) {
    if (!json_value.IsString()) {
      *error = RapidJsonMismatchedTypeError(key, "string", json_value);
      return false;
    }
    *value = json_value.GetString();
    return true;
  }
};

template <typename T>
class RapidJsonValueConverter<T, std::enable_if_t<std::is_enum<T>::value>> {
 public:
  using U = std::underlying_type_t<T>;

  template <typename Allocator>
  static rapidjson::Value From(T value, Allocator& allocator) {
    return rapidjson::Value(static_cast<U>(value));
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 T* value, std::string* error) {
    U value_tmp;
    if (!RapidJsonValueConverter<U>::To(json_value, key, &value_tmp, error))
      return false;
    *value = static_cast<T>(value_tmp);
    return true;
  }
};

template <typename T>
class RapidJsonValueConverter<std::vector<T>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const std::vector<T>& value,
                               Allocator& allocator) {
    rapidjson::Value array(rapidjson::kArrayType);
    for (size_t i = 0; i < value.size(); ++i) {
      array.PushBack(RapidJsonValueConverter<T>::From(value[i], allocator),
                     allocator);
    }
    return array;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 std::vector<T>* value, std::string* error) {
    if (!json_value.IsArray()) {
      *error = RapidJsonMismatchedTypeError(key, "array", json_value);
      return false;
    }
    std::vector<T> value_tmp;
    for (auto it = json_value.Begin(); it != json_value.End(); ++it) {
      T v;
      if (!RapidJsonValueConverter<T>::To(*it, key, &v, error)) return false;
      value_tmp.push_back(std::move(v));
    }
    *value = std::move(value_tmp);
    return true;
  }
};

template <typename T, typename Allocator>
void AddJsonElement(rapidjson::Value& json_value, std::string_view key,
                    const T& value, Allocator& allocator) {
  json_value.AddMember(rapidjson::StringRef(key.data(), key.length()),
                       RapidJsonValueConverter<T>::From(value, allocator),
                       allocator);
}

template <typename T>
bool ParseJsonElement(const rapidjson::Value& json_value, std::string_view key,
                      T* value, std::string* error) {
  return RapidJsonValueConverter<T>::To(json_value[key.data()], key, value,
                                        error);
}

}  // namespace tachyon::base

namespace rapidjson {

template <typename OutputStream, typename SourceEncoding,
          typename TargetEncoding, typename StackAllocator>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           bool value) {
  writer.Bool(value);
  return writer;
}

template <typename OutputStream, typename SourceEncoding,
          typename TargetEncoding, typename StackAllocator>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           int64_t value) {
  writer.Int64(value);
  return writer;
}

template <typename OutputStream, typename SourceEncoding,
          typename TargetEncoding, typename StackAllocator>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           uint64_t value) {
  writer.Uint64(value);
  return writer;
}

template <
    typename OutputStream, typename SourceEncoding, typename TargetEncoding,
    typename StackAllocator, typename T,
    std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value &&
                     !std::is_same<T, int64_t>::value>* = nullptr>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           T value) {
  writer.Int(value);
  return writer;
}

template <
    typename OutputStream, typename SourceEncoding, typename TargetEncoding,
    typename StackAllocator, typename T,
    std::enable_if_t<std::is_integral<T>::value && !std::is_signed<T>::value &&
                     !std::is_same<T, uint64_t>::value &&
                     !std::is_same<T, bool>::value>* = nullptr>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           T value) {
  writer.Uint(value);
  return writer;
}

template <typename OutputStream, typename SourceEncoding,
          typename TargetEncoding, typename StackAllocator, typename T,
          std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           T value) {
  writer.Double(value);
  return writer;
}

template <typename OutputStream, typename SourceEncoding,
          typename TargetEncoding, typename StackAllocator>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           std::string_view value) {
  writer.String(value.data());
  return writer;
}

}  // namespace rapidjson

#endif  // TACHYON_BASE_JSON_RAPIDJSON_UTIL_H_
