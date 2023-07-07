#include "tachyon/base/strings/string_number_conversions.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_number_conversions_internal.h"

namespace tachyon {
namespace base {

bool StringToInt(std::string_view input, int* output) {
  return absl::SimpleAtoi(input, output);
}

bool StringToUint(std::string_view input, unsigned* output) {
  return absl::SimpleAtoi(input, output);
}

bool StringToInt64(std::string_view input, int64_t* output) {
  return absl::SimpleAtoi(input, output);
}

bool StringToUint64(std::string_view input, uint64_t* output) {
  return absl::SimpleAtoi(input, output);
}

bool StringToSizeT(std::string_view input, size_t* output) {
  return absl::SimpleAtoi(input, output);
}

bool StringToFloat(std::string_view input, float* output) {
  return absl::SimpleAtof(input, output);
}

bool StringToDouble(std::string_view input, double* output) {
  return absl::SimpleAtod(input, output);
}

std::string HexEncode(const void* bytes, size_t size, bool use_lower_case) {
  static const char kUpperHexChars[] = "0123456789ABCDEF";
  static const char kLowerHexChars[] = "0123456789abcdef";

  const char* hex_chars = nullptr;
  if (use_lower_case) {
    hex_chars = kLowerHexChars;
  } else {
    hex_chars = kUpperHexChars;
  }

  // Each input byte creates two output hex characters.
  std::string ret(size * 2, '\0');

  for (size_t i = 0; i < size; ++i) {
    char b = reinterpret_cast<const char*>(bytes)[i];
    ret[(i * 2)] = hex_chars[(b >> 4) & 0xf];
    ret[(i * 2) + 1] = hex_chars[b & 0xf];
  }
  return ret;
}

std::string HexEncode(absl::Span<const uint8_t> bytes, bool use_lower_case) {
  return HexEncode(bytes.data(), bytes.size(), use_lower_case);
}

bool HexStringToInt(std::string_view input, int* output) {
  return absl::numbers_internal::safe_strtoi_base(input, output, 16);
}

bool HexStringToUint(std::string_view input, unsigned* output) {
  return absl::numbers_internal::safe_strtoi_base(input, output, 16);
}

bool HexStringToInt64(std::string_view input, int64_t* output) {
  return absl::numbers_internal::safe_strtoi_base(input, output, 16);
}

bool HexStringToUint64(std::string_view input, uint64_t* output) {
  return absl::numbers_internal::safe_strtoi_base(input, output, 16);
}

bool HexStringToBytes(std::string_view input, std::vector<uint8_t>* output) {
  DCHECK(output->empty());
  return internal::HexStringToByteContainer(input, std::back_inserter(*output));
}

bool HexStringToString(std::string_view input, std::string* output) {
  DCHECK(output->empty());
  return internal::HexStringToByteContainer(input, std::back_inserter(*output));
}

bool HexStringToSpan(std::string_view input, absl::Span<uint8_t> output) {
  if (input.size() / 2 != output.size()) return false;

  return internal::HexStringToByteContainer(input, output.begin());
}

}  // namespace base
}  // namespace tachyon
