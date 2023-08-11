// Copyright (c) 2020 The Console Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_CONSOLE_CONSOLE_STREAM_H_
#define TACHYON_BASE_CONSOLE_CONSOLE_STREAM_H_

#include <stdint.h>

#include <iostream>

#include "tachyon/base/color/color.h"
#include "tachyon/base/console/console.h"
#include "tachyon/base/console/sgr_parameters.h"
#include "tachyon/export.h"

namespace tachyon::base {

class TACHYON_EXPORT ConsoleStream {
 public:
  explicit ConsoleStream(std::ostream& ostream_ = std::cout);
  ~ConsoleStream();

#define SGR_PARAMETER_LIST(name, code) ConsoleStream& name();
#include "tachyon/base/console/sgr_parameter_list.h"
#undef SGR_PARAMETER_LIST

  ConsoleStream& Rgb(Rgba rgba);
  ConsoleStream& Rgb(uint8_t r, uint8_t g, uint8_t b);
  ConsoleStream& BgRgb(Rgba rgba);
  ConsoleStream& BgRgb(uint8_t r, uint8_t g, uint8_t b);

  // Cursor Control
  // Sets the cursor position where subsequent text will begin. If no row/column
  // parameters are provided, the cursor will move to the home
  // position, at the upper left of the screen.
  ConsoleStream& SetCursor(size_t row = 0, size_t column = 0);
  // Moves the cursor up by |n| rows.
  ConsoleStream& CursorUp(size_t n = 1);
  // Moves the cursor down by |n| rows.
  ConsoleStream& CursorDown(size_t n = 1);
  // Moves the cursor forward by |n| columns.
  ConsoleStream& CursorForward(size_t n = 1);
  // Moves the cursor backward by |n| columns.
  ConsoleStream& CursorBackward(size_t n = 1);
  // Saves current cursor position.
  ConsoleStream& SaveCursor();
  // Restores cursor position after the save point.
  ConsoleStream& RestoreCursor();
  // Save current cursor position and attributes.
  ConsoleStream& SaveCursorAndAttributes();
  // Restores cursor position after the save point and attributes.
  ConsoleStream& RestoreCursorAndAttributes();

  // Scrolling
  // Enable scrolling for entire display.
  ConsoleStream& ScrollScreen();
  // Enable scrolling from row |start| to row |end|.
  ConsoleStream& ScrollScreen(size_t start, size_t end);
  // Scroll display down one line.
  ConsoleStream& ScrollDown();
  // Scroll display up one line.
  ConsoleStream& ScrollUp();

  // Tab Control
  // Sets a tab at the current position.
  ConsoleStream& SetTab();
  // Clears tab at the current position.
  ConsoleStream& ClearTab();
  // Clears all tabs.
  ConsoleStream& ClearAllTab();

  // Erasing Text
  // Erases from the current cursor position to the end of the current line.
  ConsoleStream& EraseEndOfLine();
  // Erases from the current cursor position to the start of the current line.
  ConsoleStream& EraseStartOfLine();
  // Erases the entire current line.
  ConsoleStream& EraseEntireLine();
  // Erases the screen from the current line down to the bottom of the screen.
  ConsoleStream& EraseDown();
  // Erases the screen from the current line up to the top of the screen.
  ConsoleStream& EraseUp();
  // Erases the screen with the background colour and moves the cursor to home.
  ConsoleStream& EraseScreen();

  std::ostream& ostream() { return ostream_; }

 private:
  std::ostream& ostream_;
  Console::Info console_info_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_CONSOLE_CONSOLE_STREAM_H_
