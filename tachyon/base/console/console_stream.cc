// Copyright (c) 2020 The Console Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/console/console_stream.h"

#include "tachyon/base/color/color_conversions.h"

namespace tachyon::base {

ConsoleStream::ConsoleStream(std::ostream& ostream)
    : ostream_(ostream), console_info_(Console::GetInfo()) {}

ConsoleStream::~ConsoleStream() { Reset(); }

#define SGR_PARAMETER_LIST(name, code)   \
  ConsoleStream& ConsoleStream::name() { \
    ostream_ << sgr_params::k##name;     \
    return *this;                        \
  }
#include "tachyon/base/console/sgr_parameter_list.h"
#undef SGR_PARAMETER_LIST

ConsoleStream& ConsoleStream::Rgb(Rgba rgba) {
  return Rgb(rgba.r, rgba.g, rgba.b);
}

ConsoleStream& ConsoleStream::Rgb(uint8_t r, uint8_t g, uint8_t b) {
  if (console_info_.support_truecolor) {
    ostream_ << Rgb24(r, g, b);
  } else if (console_info_.support_8bit_color) {
    if (r == g && g == b) {
      ostream_ << Grayscale8(r * (23.0 / 255.0));
    } else {
      ostream_ << Rgb8(r, g, b);
    }
  }
  return *this;
}

ConsoleStream& ConsoleStream::BgRgb(Rgba rgba) {
  return BgRgb(rgba.r, rgba.g, rgba.b);
}

ConsoleStream& ConsoleStream::BgRgb(uint8_t r, uint8_t g, uint8_t b) {
  if (console_info_.support_truecolor) {
    ostream_ << BgRgb24(r, g, b);
  } else if (console_info_.support_8bit_color) {
    if (r == g && g == b) {
      ostream_ << BgGrayscale8(r * (23.0 / 255.0));
    } else {
      ostream_ << BgRgb8(r, g, b);
    }
  }
  return *this;
}

ConsoleStream& ConsoleStream::SetCursor(size_t row, size_t column) {
  ostream_ << "\e[" << row << ";" << column << "H";
  return *this;
}

ConsoleStream& ConsoleStream::CursorUp(size_t n) {
  ostream_ << "\e[" << n << "A";
  return *this;
}

ConsoleStream& ConsoleStream::CursorDown(size_t n) {
  ostream_ << "\e[" << n << "B";
  return *this;
}

ConsoleStream& ConsoleStream::CursorForward(size_t n) {
  ostream_ << "\e[" << n << "C";
  return *this;
}

ConsoleStream& ConsoleStream::CursorBackward(size_t n) {
  ostream_ << "\e[" << n << "D";
  return *this;
}

ConsoleStream& ConsoleStream::SaveCursor() {
  ostream_ << "\e[s";
  return *this;
}

ConsoleStream& ConsoleStream::RestoreCursor() {
  ostream_ << "\e[u";
  return *this;
}

ConsoleStream& ConsoleStream::SaveCursorAndAttributes() {
  ostream_ << "\e7";
  return *this;
}

ConsoleStream& ConsoleStream::RestoreCursorAndAttributes() {
  ostream_ << "\e8";
  return *this;
}

ConsoleStream& ConsoleStream::ScrollScreen() {
  ostream_ << "\e[r";
  return *this;
}

ConsoleStream& ConsoleStream::ScrollScreen(size_t start, size_t end) {
  ostream_ << "\e[" << start << ";" << end << "r";
  return *this;
}

ConsoleStream& ConsoleStream::ScrollDown() {
  ostream_ << "\eD";
  return *this;
}

ConsoleStream& ConsoleStream::ScrollUp() {
  ostream_ << "\eM";
  return *this;
}

ConsoleStream& ConsoleStream::SetTab() {
  ostream_ << "\eH";
  return *this;
}

ConsoleStream& ConsoleStream::ClearTab() {
  ostream_ << "\e[g";
  return *this;
}

ConsoleStream& ConsoleStream::ClearAllTab() {
  ostream_ << "\e[3g";
  return *this;
}

ConsoleStream& ConsoleStream::EraseEndOfLine() {
  ostream_ << "\e[K";
  return *this;
}

ConsoleStream& ConsoleStream::EraseStartOfLine() {
  ostream_ << "\e[1K";
  return *this;
}

ConsoleStream& ConsoleStream::EraseEntireLine() {
  ostream_ << "\e[2K";
  return *this;
}

ConsoleStream& ConsoleStream::EraseDown() {
  ostream_ << "\e[J";
  return *this;
}

ConsoleStream& ConsoleStream::EraseUp() {
  ostream_ << "\e[1J";
  return *this;
}

ConsoleStream& ConsoleStream::EraseScreen() {
  ostream_ << "\e[2J";
  return *this;
}

}  // namespace tachyon::base
