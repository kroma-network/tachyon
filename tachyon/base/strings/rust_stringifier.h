#ifndef TACHYON_BASE_STRINGS_RUST_STRINGIFIER_H_
#define TACHYON_BASE_STRINGS_RUST_STRINGIFIER_H_

#include <optional>
#include <sstream>
#include <string>
#include <type_traits>

#include "tachyon/export.h"

namespace tachyon::base {

class RustFormatter;

namespace internal {

template <typename T, typename SFINAE = void>
class RustDebugStringifier;

class TACHYON_EXPORT DebugStruct {
 public:
  DebugStruct(RustFormatter* fmt, std::string_view name) : fmt_(fmt) {
    ss_ << name << " { ";
  }

  template <typename T>
  DebugStruct& Field(std::string_view name, const T& value) {
    if (has_field_) {
      ss_ << ", ";
    }
    ss_ << name << ": ";
    RustDebugStringifier<T>::AppendToStream(ss_, *fmt_, value);
    has_field_ = true;
    return *this;
  }

  std::string Finish() {
    ss_ << " }";
    return ss_.str();
  }

 private:
  RustFormatter* fmt_ = nullptr;
  std::stringstream ss_;
  bool has_field_ = false;
};

}  // namespace internal

class TACHYON_EXPORT RustFormatter {
 public:
  RustFormatter() = default;

  internal::DebugStruct DebugStruct(std::string_view name) {
    return internal::DebugStruct(this, name);
  }
};

namespace internal {

template <>
class RustDebugStringifier<bool> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      bool value) {
    if (value) {
      return os << "true";
    } else {
      return os << "false";
    }
  }
};

template <typename T>
class RustDebugStringifier<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      T value) {
    return os << value;
  }
};

template <typename CharTy>
class RustDebugStringifier<std::basic_string_view<CharTy>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      std::basic_string_view<CharTy> value) {
    return os << "\"" << value << "\"";
  }
};

template <typename CharTy>
class RustDebugStringifier<std::basic_string<CharTy>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      const std::basic_string<CharTy>& value) {
    return os << "\"" << value << "\"";
  }
};

template <typename CharTy>
class RustDebugStringifier<
    const CharTy*,
    std::enable_if_t<
        std::is_same_v<CharTy, char> || std::is_same_v<CharTy, wchar_t> ||
        std::is_same_v<CharTy, char16_t> || std::is_same_v<CharTy, char32_t>>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      const CharTy* value) {
    return os << "\"" << value << "\"";
  }
};

template <typename T>
class RustDebugStringifier<std::optional<T>> {
 public:
  static std::ostream& AppendToStream(std::ostream& os, RustFormatter& fmt,
                                      const std::optional<T>& value) {
    if (value.has_value()) {
      os << "Some(";
      RustDebugStringifier<T>::AppendToStream(os, fmt, value.value());
      return os << ")";
    } else {
      return os << "None";
    }
  }
};

}  // namespace internal

template <typename T>
std::string ToRustDebugString(RustFormatter& fmt, const T& value) {
  std::stringstream ss;
  internal::RustDebugStringifier<T>::AppendToStream(ss, fmt, value);
  return ss.str();
}

template <typename T>
std::string ToRustDebugString(const T& value) {
  RustFormatter fmt;
  return ToRustDebugString(fmt, value);
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_STRINGS_RUST_STRINGIFIER_H_
