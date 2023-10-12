// Copyright (c) 2019 The Color Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_COLOR_COLOR_H_
#define TACHYON_BASE_COLOR_COLOR_H_

#include <stdint.h>

#include <array>
#include <string>

#include "tachyon/export.h"

namespace tachyon::base {
struct TACHYON_EXPORT RgbaIndexes {
  int r;
  int g;
  int b;
  int a;
};

TACHYON_EXPORT extern const RgbaIndexes kRgbIndexes;
TACHYON_EXPORT extern const RgbaIndexes kRgbaIndexes;
TACHYON_EXPORT extern const RgbaIndexes kBgrIndexes;
TACHYON_EXPORT extern const RgbaIndexes kBgraIndexes;
TACHYON_EXPORT extern const RgbaIndexes kArgbIndexes;

// Rgba class is copyable, assignable, and occupy 32-bits per instance.
// As a result, prefer passing them by value:
//   void MyFunction(Rgba arg);
// If circumstances require, you may also pass by const reference:
//   void MyFunction(const Rgba& arg);  // Not preferred.
struct TACHYON_EXPORT Rgba {
  constexpr Rgba() : Rgba(0, 0, 0, 0) {}
  constexpr Rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
      : r(r), g(g), b(b), a(a) {}
  constexpr explicit Rgba(uint32_t rgba) : rgba(rgba) {}
  constexpr explicit Rgba(const uint8_t* data)
      : Rgba(data[0], data[1], data[2], data[3]) {}
  // This is convenience constructor.
  // e.g) Without this, Rgba(0, 0, 0) makes you annoyed!
  constexpr Rgba(int r, int g, int b, int a = 255)
      : Rgba(static_cast<uint8_t>(r), static_cast<uint8_t>(g),
             static_cast<uint8_t>(b), static_cast<uint8_t>(a)) {}
  // |r|, |g|, |b| and |a| should be in 0 to 1.
  constexpr Rgba(float r, float g, float b, float a = 1)
      : Rgba(static_cast<uint8_t>(r * 255), static_cast<uint8_t>(g * 255),
             static_cast<uint8_t>(b * 255), static_cast<uint8_t>(a * 255)) {}
  // This is convenience constructor.
  // e.g) Without this, Rgba(0.0, 0.0, 0.0) makes you annoyed!
  constexpr Rgba(double r, double g, double b, double a = 1)
      : Rgba(static_cast<float>(r), static_cast<float>(g),
             static_cast<float>(b), static_cast<float>(a)) {}
  constexpr explicit Rgba(const float* data)
      : Rgba(data[0], data[1], data[2], data[3]) {}

  // Converts to std::string. It's redirected to ToRgbaString().
  std::string ToString() const;
  // Converts to std::string.
  // e.g) if (r, g, b, a) is (0, 0, 0, 1), then it returns "rgb(0, 0, 0)".
  std::string ToRgbString() const;
  // Converts to std::string.
  // e.g) if (r, g, b, a) is (0, 0, 0, 1) and |hex| is false, then it returns
  // "rgba(0, 0, 0, 1)". Or |hex| is true, it returns "#00000001".
  std::string ToRgbaString() const;
  // Converts to std::string.
  // e.g) if (r, g, b, a) is (255, 0, 0, 1), it returns "#ff0000".
  std::string ToRgbHexString() const;
  // Converts to std::string.
  // e.g) if (r, g, b, a) is (255, 0, 0, 1), it returns "#ff000001".
  std::string ToRgbaHexString() const;
  // Converts from std::string. Returns true if succeeded.
  // Expected form is "rgb(r, g, b)", "rgba(r, g, b, a), #abcdef or "#ABCDEF".
  bool FromString(const std::string& text);

  std::array<float, 3> ToFloatingArray3() const {
    return {r / 255.f, g / 255.f, b / 255.f};
  }

  std::array<float, 4> ToFloatingArray4() const {
    return {r / 255.f, g / 255.f, b / 255.f, a / 255.f};
  }

  constexpr bool IsValid() const { return true; }

  constexpr bool IsTransparent() const { return a != 255; }

  constexpr bool IsOpaque() const { return a == 255; }

  Rgba Swap(const RgbaIndexes& rgba_indexes) const;

  constexpr Rgba PremultipliedAlpha() const {
    return {static_cast<uint8_t>(static_cast<float>(r) * a / 255.f),
            static_cast<uint8_t>(static_cast<float>(g) * a / 255.f),
            static_cast<uint8_t>(static_cast<float>(b) * a / 255.f), a};
  }

  union {
    struct {
      uint8_t r;
      uint8_t g;
      uint8_t b;
      uint8_t a;
    };
    uint8_t array[4];
    uint32_t rgba;
  };
};

inline bool operator==(Rgba rgb, Rgba rgb2) { return rgb.rgba == rgb2.rgba; }

inline bool operator!=(Rgba rgb, Rgba rgb2) { return !operator==(rgb, rgb2); }

TACHYON_EXPORT std::ostream& operator<<(std::ostream& os, Rgba rgb);

struct TACHYON_EXPORT Hsv {
  constexpr Hsv() : Hsv(0, 0, 0, 1) {}
  constexpr Hsv(float h, float s, float v, float a = 1)
      : h(h), s(s), v(v), a(a) {}
  constexpr explicit Hsv(const float* data)
      : Hsv(data[0], data[1], data[2], data[3]) {}

  Hsv(const Hsv& other) = default;
  Hsv& operator=(const Hsv& other) = default;

  // Converts to std::string. It's redirected to ToHsvaString().
  std::string ToString() const;
  // Converts to std::string.
  // e.g) if (h, s, v, a) is (0, 0, 0, 1), then it returns "hsv(0, 0, 0)".
  std::string ToHsvString() const;
  // e.g) if (h, s, v, a) is (0, 0, 0, 1), then it returns "hsva(0, 0, 0, 1)".
  std::string ToHsvaString() const;
  // Converts from std::string. Returns true if succeeded.
  // Expected form is "hsv(h, s, v)" or "hsva(h, s, v, a)".
  bool FromString(const std::string& text);

  constexpr bool IsValid() const {
    return h >= 0 && h <= 360 && s >= 0 && s <= 1 && v >= 0 && v <= 1 &&
           a >= 0 && a <= 1;
  }

  constexpr bool IsTransparent() const { return a != 1; }

  constexpr bool IsOpaque() const { return a == 1; }

  union {
    struct {
      float h;
      float s;
      float v;
      float a;
    };
    float array[4];
  };
};

inline bool operator==(const Hsv& hsv, const Hsv& hsv2) {
  return hsv.h == hsv2.h && hsv.s == hsv2.s && hsv.v == hsv2.v &&
         hsv.a == hsv2.a;
}

inline bool operator!=(const Hsv& hsv, const Hsv& hsv2) {
  return !operator==(hsv, hsv2);
}

TACHYON_EXPORT std::ostream& operator<<(std::ostream& os, const Hsv& hsv);

}  // namespace tachyon::base

#endif  // TACHYON_BASE_COLOR_COLOR_H_
