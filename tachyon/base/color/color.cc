// Copyright (c) 2019 The Color Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/color/color.h"

#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"

#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/base/strings/string_util.h"

namespace tachyon::base {
namespace {

// 0 ~ 255
template <typename T,
          std::enable_if_t<std::is_same<T, uint8_t>::value>* = nullptr>
bool StringToNumber(std::string_view input, uint8_t* number) {
  unsigned n;
  if (!StringToUint(input, &n)) return false;
  if (n > 255) return false;
  *number = static_cast<uint8_t>(n);
  return true;
}

template <typename T,
          std::enable_if_t<std::is_same<T, float>::value>* = nullptr>
bool StringToNumber(std::string_view input, float* number) {
  float n;
  if (!StringToFloat(input, &n)) return false;
  if (n > 1) return false;
  *number = n;
  return true;
}

// Parse comma separated numbers.
template <size_t N, typename T>
bool ParseCommaSeparatedNumbers(std::string_view* input, T* numbers) {
  std::vector<std::string_view> v =
      absl::StrSplit(*input, ',', absl::SkipWhitespace());
  if (v.size() != N) return false;
  T n_temps[N];
  for (size_t i = 0; i < N; ++i) {
    if (!StringToNumber<T>(v[i], &n_temps[i])) return false;
  }
  for (size_t i = 0; i < N; ++i) {
    numbers[i] = n_temps[i];
  }
  return true;
}

bool ParseRgbNumbers(std::string_view* input, uint8_t* numbers) {
  return ParseCommaSeparatedNumbers<3>(input, numbers);
}

bool ParseRgbaNumbers(std::string_view* input, uint8_t* numbers) {
  return ParseCommaSeparatedNumbers<4>(input, numbers);
}

bool ParseHsvNumbers(std::string_view* input, float* numbers) {
  Hsv hsv;
  if (!ParseCommaSeparatedNumbers<3>(input, hsv.array)) return false;
  if (!hsv.IsValid()) return false;
  memcpy(numbers, hsv.array, sizeof(float) * 3);
  return true;
}

bool ParseHsvaNumbers(std::string_view* input, float* numbers) {
  Hsv hsv;
  if (!ParseCommaSeparatedNumbers<4>(input, hsv.array)) return false;
  if (!hsv.IsValid()) return false;
  memcpy(numbers, hsv.array, sizeof(float) * 4);
  return true;
}

bool ConsumeHex(std::string_view* input, uint8_t* c) {
  std::string_view c_sv(input->data(), 2);
  uint32_t n;
  if (!HexStringToUint(c_sv, &n)) return false;
  if (n > 255) return false;
  *c = static_cast<uint8_t>(n);
  input->remove_prefix(2);
  return true;
}

bool ConsumeHexRgba(std::string_view* input, uint8_t* r, uint8_t* g, uint8_t* b,
                    uint8_t* a) {
  uint8_t r_temp;
  uint8_t g_temp;
  uint8_t b_temp;
  uint8_t a_temp = 255;
  size_t len = input->length();
  if (len == 6 || len == 8) {
    if (!(ConsumeHex(input, &r_temp) && ConsumeHex(input, &g_temp) &&
          ConsumeHex(input, &b_temp)))
      return false;
    if (len == 8) {
      if (!ConsumeHex(input, &a_temp)) return false;
    }
  } else {
    return false;
  }

  *r = r_temp;
  *g = g_temp;
  *b = b_temp;
  *a = a_temp;

  return true;
}

}  // namespace

const RgbaIndexes kRgbIndexes{0, 1, 2, -1};
const RgbaIndexes kRgbaIndexes{0, 1, 2, 3};
const RgbaIndexes kBgrIndexes{2, 1, 0, -1};
const RgbaIndexes kBgraIndexes{2, 1, 0, 3};
const RgbaIndexes kArgbIndexes{1, 2, 3, 0};

std::string Rgba::ToString() const { return ToRgbaString(); }

std::string Rgba::ToRgbString() const {
  return absl::StrFormat("rgb(%u, %u, %u)", r, g, b);
}

std::string Rgba::ToRgbaString() const {
  return absl::StrFormat("rgba(%u, %u, %u, %u)", r, g, b, a);
}

std::string Rgba::ToRgbHexString() const {
  return absl::StrFormat("#%02x%02x%02x", r, g, b);
}

std::string Rgba::ToRgbaHexString() const {
  return absl::StrFormat("#%02x%02x%02x%02x", r, g, b, a);
}

bool Rgba::FromString(const std::string& text) {
  std::string_view input(text);
  if (ConsumePrefix(&input, "rgb(") && ConsumeSuffix(&input, ")")) {
    if (ParseRgbNumbers(&input, array)) {
      a = 255;
      return true;
    }
  } else if (ConsumePrefix(&input, "rgba(") && ConsumeSuffix(&input, ")")) {
    return ParseRgbaNumbers(&input, array);
  } else if (ConsumePrefix(&input, "#")) {
    return ConsumeHexRgba(&input, &r, &g, &b, &a);
  }

  return false;
}

Rgba Rgba::Swap(const RgbaIndexes& rgba_indexes) const {
  Rgba swapped;
  swapped.array[rgba_indexes.r] = r;
  swapped.array[rgba_indexes.g] = g;
  swapped.array[rgba_indexes.b] = b;
  if (rgba_indexes.a != -1) {
    swapped.array[rgba_indexes.a] = a;
  } else {
    swapped.array[rgba_indexes.a] = 255;
  }
  return swapped;
}

std::string Hsv::ToString() const { return ToHsvaString(); }

std::string Hsv::ToHsvString() const {
  return absl::StrFormat("hsv(%f, %f, %f)", h, s, v);
}

std::string Hsv::ToHsvaString() const {
  return absl::StrFormat("hsva(%f, %f, %f, %f)", h, s, v, a);
}

bool Hsv::FromString(const std::string& text) {
  std::string_view input(text);
  if (ConsumePrefix(&input, "hsv(") && ConsumeSuffix(&input, ")")) {
    if (ParseHsvNumbers(&input, array)) {
      a = 1;
      return true;
    }
  } else if (ConsumePrefix(&input, "hsva(") && ConsumeSuffix(&input, ")")) {
    return ParseHsvaNumbers(&input, array);
  } else {
    return false;
  }

  return false;
}

}  // namespace tachyon::base
