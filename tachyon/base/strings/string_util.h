#ifndef TACHYON_BASE_STRINGS_STRING_UTIL_H_
#define TACHYON_BASE_STRINGS_STRING_UTIL_H_

#include <string>
#include <string_view>

#include "tachyon/export.h"

namespace tachyon::base {

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

}  // namespace tachyon::base

#endif  // TACHYON_BASE_STRINGS_STRING_UTIL_H_
