#ifndef TACHYON_BASE_STRINGS_STRING_UTIL_H_
#define TACHYON_BASE_STRINGS_STRING_UTIL_H_

#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/strings/string_util_internal.h"
#include "tachyon/export.h"

namespace tachyon::base {

// ASCII-specific tolower.  The standard library's tolower is locale sensitive,
// so we don't want to use it here.
template <typename CharT,
          typename = std::enable_if_t<std::is_integral<CharT>::value>>
constexpr CharT ToLowerASCII(CharT c) {
  return internal::ToLowerASCII(c);
}

// ASCII-specific toupper.  The standard library's toupper is locale sensitive,
// so we don't want to use it here.
template <typename CharT,
          typename = std::enable_if_t<std::is_integral<CharT>::value>>
CharT ToUpperASCII(CharT c) {
  return (c >= 'a' && c <= 'z') ? static_cast<CharT>(c + 'A' - 'a') : c;
}

// Converts the given string to its ASCII-lowercase equivalent. Non-ASCII
// bytes (or UTF-16 code units in `std::u16string_view`) are permitted but will
// be unmodified.
TACHYON_EXPORT std::string ToLowerASCII(std::string_view str);
TACHYON_EXPORT std::u16string ToLowerASCII(std::u16string_view str);

// Converts the given string to its ASCII-uppercase equivalent. Non-ASCII
// bytes (or UTF-16 code units in `std::u16string_view`) are permitted but will
// be unmodified.
TACHYON_EXPORT std::string ToUpperASCII(std::string_view str);
TACHYON_EXPORT std::u16string ToUpperASCII(std::u16string_view str);

// Like strcasecmp for ASCII case-insensitive comparisons only. Returns:
//   -1  (a < b)
//    0  (a == b)
//    1  (a > b)
// (unlike strcasecmp which can return values greater or less than 1/-1). To
// compare all Unicode code points case-insensitively, use base::i18n::ToLower
// or base::i18n::FoldCase and then just call the normal string operators on the
// result.
//
// Non-ASCII bytes (or UTF-16 code units in `std::u16string_view`) are permitted
// but will be compared unmodified.
TACHYON_EXPORT constexpr int CompareCaseInsensitiveASCII(std::string_view a,
                                                         std::string_view b) {
  return internal::CompareCaseInsensitiveASCIIT(a, b);
}

// Equality for ASCII case-insensitive comparisons. Non-ASCII bytes (or UTF-16
// code units in `std::u16string_view`) are permitted but will be compared
// unmodified.
inline bool EqualsCaseInsensitiveASCII(std::string_view a, std::string_view b) {
  return internal::EqualsCaseInsensitiveASCIIT(a, b);
}

// These threadsafe functions return references to globally unique empty
// strings.
//
// It is likely faster to construct a new empty string object (just a few
// instructions to set the length to 0) than to get the empty string instance
// returned by these functions (which requires threadsafe static access).
//
// Therefore, DO NOT USE THESE AS A GENERAL-PURPOSE SUBSTITUTE FOR DEFAULT
// CONSTRUCTORS. There is only one case where you should use these: functions
// which need to return a string by reference (e.g. as a class member
// accessor), and don't have an empty string to use (e.g. in an error case).
// These should not be used as initializers, function arguments, or return
// values for functions which return by value or outparam.
TACHYON_EXPORT const std::string& EmptyString();

// Indicates case sensitivity of comparisons. Only ASCII case insensitivity
// is supported. Full Unicode case-insensitive conversions would need ICU
// and it's not supported yet.
enum class CompareCase {
  SENSITIVE,
  INSENSITIVE_ASCII,
};

TACHYON_EXPORT bool StartsWith(
    std::string_view str, std::string_view search_for,
    CompareCase case_sensitivity = CompareCase::SENSITIVE);

TACHYON_EXPORT bool EndsWith(
    std::string_view str, std::string_view search_for,
    CompareCase case_sensitivity = CompareCase::SENSITIVE);

TACHYON_EXPORT bool ConsumePrefix(
    std::string_view* str, std::string_view search_for,
    CompareCase case_sensitivity = CompareCase::SENSITIVE);

TACHYON_EXPORT bool ConsumeSuffix(
    std::string_view* str, std::string_view search_for,
    CompareCase case_sensitivity = CompareCase::SENSITIVE);

TACHYON_EXPORT bool ConsumePrefix0x(std::string_view* str);
TACHYON_EXPORT std::string MaybePrepend0x(std::string_view str);
TACHYON_EXPORT std::string MaybePrepend0x(std::string&& str);
TACHYON_EXPORT std::string MaybePrepend0x(const std::string& str);

ALWAYS_INLINE const char* BoolToString(bool b) { return b ? "true" : "false"; }

template <typename T>
std::string VectorToString(const std::vector<T>& data) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < data.size(); ++i) {
    ss << data[i];
    if (i != data.size() - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

template <typename T>
std::string Vector2DToString(const std::vector<std::vector<T>>& data) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < data.size(); ++i) {
    ss << VectorToString(data[i]);
    if (i != data.size() - 1) {
      ss << ",";
    }
  }
  ss << "]";
  return ss.str();
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_STRINGS_STRING_UTIL_H_
