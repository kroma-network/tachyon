#ifndef TACHYON_STRINGS_STRING_NUMBER_CONVERSIONS_H_
#define TACHYON_STRINGS_STRING_NUMBER_CONVERSIONS_H_

#include <string>
#include <string_view>
#include <type_traits>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

#include "tachyon/export.h"

namespace tachyon::base {

template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
std::string NumberToString(T value) {
  return absl::StrCat(value);
}

TACHYON_EXPORT bool StringToInt(std::string_view input, int* output);
TACHYON_EXPORT bool StringToUint(std::string_view input, unsigned* output);
TACHYON_EXPORT bool StringToInt64(std::string_view input, int64_t* output);
TACHYON_EXPORT bool StringToUint64(std::string_view input, uint64_t* output);
TACHYON_EXPORT bool StringToSizeT(std::string_view input, size_t* output);
TACHYON_EXPORT bool StringToFloat(std::string_view input, float* output);
TACHYON_EXPORT bool StringToDouble(std::string_view input, double* output);

template <typename T, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
std::string HexToString(T value) {
  return absl::StrCat(absl::Hex(value));
}

TACHYON_EXPORT std::string HexEncode(const void* bytes, size_t size,
                                     bool use_lower_case);
TACHYON_EXPORT std::string HexEncode(absl::Span<const uint8_t> bytes,
                                     bool use_lower_case);

TACHYON_EXPORT bool HexStringToInt(std::string_view input, int* output);
TACHYON_EXPORT bool HexStringToUint(std::string_view input, unsigned* output);
TACHYON_EXPORT bool HexStringToInt64(std::string_view input, int64_t* output);
TACHYON_EXPORT bool HexStringToUint64(std::string_view input, uint64_t* output);

// Similar to the previous functions, except that output is a vector of bytes.
// |*output| will contain as many bytes as were successfully parsed prior to the
// error.  There is no overflow, but input.size() must be evenly divisible by 2.
// Leading 0x or +/- are not allowed.
TACHYON_EXPORT bool HexStringToBytes(std::string_view input,
                                     std::vector<uint8_t>* output);

// Same as HexStringToBytes, but for an std::string.
TACHYON_EXPORT bool HexStringToString(std::string_view input,
                                      std::string* output);

// Decodes the hex string |input| into a presized |output|. The output buffer
// must be sized exactly to |input.size() / 2| or decoding will fail and no
// bytes will be written to |output|. Decoding an empty input is also
// considered a failure. When decoding fails due to encountering invalid input
// characters, |output| will have been filled with the decoded bytes up until
// the failure.
TACHYON_EXPORT bool HexStringToSpan(std::string_view input,
                                    absl::Span<uint8_t> output);

}  // namespace tachyon::base

#endif  // TACHYON_STRINGS_STRING_NUMBER_CONVERSIONS_H_
