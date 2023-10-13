// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_STRINGS_STRING_NUMBER_CONVERSIONS_INTERNAL_H_
#define TACHYON_BASE_STRINGS_STRING_NUMBER_CONVERSIONS_INTERNAL_H_

#include <stdint.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string_view>

namespace tachyon::base {
namespace internal {

// Utility to convert a character to a digit in a given base
template <int BASE, typename CHAR>
std::optional<uint8_t> CharToDigit(CHAR c) {
  static_assert(1 <= BASE && BASE <= 36, "BASE needs to be in [1, 36]");
  if (c >= '0' && c < '0' + std::min(BASE, 10)) return c - '0';

  if (c >= 'a' && c < 'a' + BASE - 10) return c - 'a' + 10;

  if (c >= 'A' && c < 'A' + BASE - 10) return c - 'A' + 10;

  return std::nullopt;
}

template <typename OutIter>
static bool HexStringToByteContainer(std::string_view input, OutIter output) {
  size_t count = input.size();
  if (count == 0 || (count % 2) != 0) return false;
  for (uintptr_t i = 0; i < count / 2; ++i) {
    // most significant 4 bits
    std::optional<uint8_t> msb = CharToDigit<16>(input[i * 2]);
    // least significant 4 bits
    std::optional<uint8_t> lsb = CharToDigit<16>(input[i * 2 + 1]);
    if (!msb || !lsb) {
      return false;
    }
    *(output++) = (*msb << 4) | *lsb;
  }
  return true;
}

}  // namespace internal
}  // namespace tachyon::base

#endif  // TACHYON_BASE_STRINGS_STRING_NUMBER_CONVERSIONS_INTERNAL_H_
