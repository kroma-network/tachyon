// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// clang-format off

#ifndef TACHYON_BASE_STRINGS_STRING_UTIL_IMPL_HELPERS_H_
#define TACHYON_BASE_STRINGS_STRING_UTIL_IMPL_HELPERS_H_

#include <string>

namespace tachyon::base::internal {

template <typename T, typename CharT = typename T::value_type>
std::basic_string<CharT> ToLowerASCIIImpl(T str) {
  std::basic_string<CharT> ret;
  ret.reserve(str.size());
  for (size_t i = 0; i < str.size(); i++)
    ret.push_back(ToLowerASCII(str[i]));
  return ret;
}

template <typename T, typename CharT = typename T::value_type>
std::basic_string<CharT> ToUpperASCIIImpl(T str) {
  std::basic_string<CharT> ret;
  ret.reserve(str.size());
  for (size_t i = 0; i < str.size(); i++)
    ret.push_back(ToUpperASCII(str[i]));
  return ret;
}

}  // namespace tachyon::base::internal

#endif  // TACHYON_BASE_STRINGS_STRING_UTIL_IMPL_HELPERS_H_

// clang-format on
