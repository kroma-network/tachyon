// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// clang-format off

#ifndef TACHYON_BASE_STRINGS_STRING_UTIL_INTERNAL_H_
#define TACHYON_BASE_STRINGS_STRING_UTIL_INTERNAL_H_

#include <type_traits>
#include <string_view>

#include "tachyon/base/ranges/algorithm.h"

namespace tachyon::base::internal {

// ASCII-specific tolower.  The standard library's tolower is locale sensitive,
// so we don't want to use it here.
template <typename CharT,
          typename = std::enable_if_t<std::is_integral<CharT>::value>>
constexpr CharT ToLowerASCII(CharT c) {
  return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c;
}

template <typename T>
constexpr int CompareCaseInsensitiveASCIIT(T a, T b) {
  // Find the first characters that aren't equal and compare them.  If the end
  // of one of the strings is found before a nonequal character, the lengths
  // of the strings are compared. Compare using the unsigned type so the sort
  // order is independent of the signedness of `char`.
  static_assert(std::is_integral_v<typename T::value_type>);
  using UCharT = std::make_unsigned_t<typename T::value_type>;
  size_t i = 0;
  while (i < a.length() && i < b.length()) {
    UCharT lower_a = static_cast<UCharT>(ToLowerASCII(a[i]));
    UCharT lower_b = static_cast<UCharT>(ToLowerASCII(b[i]));
    if (lower_a < lower_b)
      return -1;
    if (lower_a > lower_b)
      return 1;
    i++;
  }

  // End of one string hit before finding a different character. Expect the
  // common case to be "strings equal" at this point so check that first.
  if (a.length() == b.length())
    return 0;

  if (a.length() < b.length())
    return -1;
  return 1;
}

template <typename CharT, typename CharU>
inline bool EqualsCaseInsensitiveASCIIT(std::basic_string_view<CharT> a,
                                        std::basic_string_view<CharU> b) {
  return ranges::equal(a, b, [](auto lhs, auto rhs) {
    return ToLowerASCII(lhs) == ToLowerASCII(rhs);
  });
}

}  // namespace tachyon::base::internal

#endif  // TACHYON_BASE_STRINGS_STRING_UTIL_INTERNAL_H_

// clang-format on
