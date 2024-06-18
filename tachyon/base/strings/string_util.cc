#include "tachyon/base/strings/string_util.h"

#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/no_destructor.h"
#include "tachyon/base/strings/string_util_impl_helpers.h"

namespace tachyon::base {
namespace {

constexpr const char* k0x = "0x";

// Functor for case-insensitive ASCII comparisons for STL algorithms like
// std::search.
struct CaseInsensitiveCompareASCII {
 public:
  bool operator()(char x, char y) const {
    return absl::ascii_tolower(x) == absl::ascii_tolower(y);
  }
};

template <typename T>
std::string DoMaybePrepend0x(T&& str) {
  if (StartsWith(str, k0x)) {
    return std::string(std::forward<T>(str));
  }
  return absl::StrCat(k0x, str);
}

}  // namespace

std::string ToLowerASCII(std::string_view str) {
  return internal::ToLowerASCIIImpl(str);
}

std::u16string ToLowerASCII(std::u16string_view str) {
  return internal::ToLowerASCIIImpl(str);
}

std::string ToUpperASCII(std::string_view str) {
  return internal::ToUpperASCIIImpl(str);
}

std::u16string ToUpperASCII(std::u16string_view str) {
  return internal::ToUpperASCIIImpl(str);
}

std::string CapitalizeASCII(std::string_view str) {
  std::string ret = std::string(str);
  if (!str.empty()) ret[0] = ToUpperASCII(str[0]);
  return ret;
}

std::u16string CapitalizeASCII(std::u16string_view str) {
  std::u16string ret = std::u16string(str);
  if (!str.empty()) ret[0] = ToUpperASCII(str[0]);
  return ret;
}

const std::string& EmptyString() {
  static const NoDestructor<std::string> s;
  return *s;
}

bool StartsWith(std::string_view str, std::string_view search_for,
                CompareCase case_sensitivity) {
  if (search_for.size() > str.size()) return false;

  std::string_view source = str.substr(0, search_for.size());

  switch (case_sensitivity) {
    case CompareCase::SENSITIVE:
      return source == search_for;

    case CompareCase::INSENSITIVE_ASCII:
      return std::equal(search_for.begin(), search_for.end(), source.begin(),
                        CaseInsensitiveCompareASCII());

    default:
      NOTREACHED();
      return false;
  }
}

bool EndsWith(std::string_view str, std::string_view search_for,
              CompareCase case_sensitivity) {
  if (search_for.size() > str.size()) return false;

  std::basic_string_view source =
      str.substr(str.size() - search_for.size(), search_for.size());

  switch (case_sensitivity) {
    case CompareCase::SENSITIVE:
      return source == search_for;

    case CompareCase::INSENSITIVE_ASCII:
      return std::equal(source.begin(), source.end(), search_for.begin(),
                        CaseInsensitiveCompareASCII());

    default:
      NOTREACHED();
      return false;
  }
}

bool ConsumePrefix(std::string_view* str, std::string_view search_for,
                   CompareCase case_sensitivity) {
  if (StartsWith(*str, search_for, case_sensitivity)) {
    str->remove_prefix(search_for.size());
    return true;
  }
  return false;
}

bool ConsumeSuffix(std::string_view* str, std::string_view search_for,
                   CompareCase case_sensitivity) {
  if (EndsWith(*str, search_for, case_sensitivity)) {
    str->remove_suffix(search_for.size());
    return true;
  }
  return false;
}

bool ConsumePrefix0x(std::string_view* str) { return ConsumePrefix(str, k0x); }

std::string MaybePrepend0x(std::string_view str) {
  return DoMaybePrepend0x(str);
}

std::string MaybePrepend0x(const std::string& str) {
  return DoMaybePrepend0x(str);
}

std::string MaybePrepend0x(std::string&& str) {
  return DoMaybePrepend0x(std::move(str));
}

std::string ToHexStringWithLeadingZero(const std::string& str, size_t num) {
  CHECK_GE(num, str.size());
  std::string_view sv = str;
  if (ConsumePrefix0x(&sv)) {
    return absl::StrCat(k0x, std::string(num - sv.size(), '0'), sv);
  } else {
    return absl::StrCat(std::string(num - sv.size(), '0'), sv);
  }
}

}  // namespace tachyon::base
