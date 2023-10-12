// Copyright (c) 2020 The Console Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/console/console.h"

#include <unistd.h>

#include <algorithm>
#include <iostream>

#include "tachyon/base/environment.h"
#include "tachyon/base/strings/string_util.h"

namespace tachyon::base {

Console::Info::Info() { Init(); }

void Console::Info::Init() {
  const char* kTerms[] = {
      "ansi",  "bvterm", "color",   "console", "cygwin", "konsole",
      "linux", "putty",  "scoansi", "screen",  "tmux",
  };

  const char* kTermsPrefix[] = {
      "eterm", "kterm", "vt100", "vt102", "vt220", "vt320", "xterm",
  };

  std::string_view term;
  if (!Environment::Get("TERM", &term)) return;
  support_ansi = std::any_of(std::begin(kTerms), std::end(kTerms),
                             [term](const char* t) { return term == t; }) ||
                 std::any_of(std::begin(kTermsPrefix), std::end(kTermsPrefix),
                             [term](const char* prefix) {
                               return StartsWith(term, prefix);
                             });

  std::string_view colorterm;
  if (!Environment::Get("COLORTERM", &colorterm)) return;
  if (colorterm == "truecolor") {
    support_truecolor = true;
  } else if (EndsWith(colorterm, "-256")) {
    support_8bit_color = true;
  }
}

// static
Console::Info& Console::GetInfo() {
  static Console::Info info;
  return info;
}

// This was taken and modified from
// LICENSE: undefined
// URL:  https://github.com/agauniyal/rang/blob/master/include/rang.hpp
// static
bool Console::IsConnected(std::ostream& os) {
  std::streambuf* osbuf = os.rdbuf();
  if (osbuf == std::cout.rdbuf()) {
    static const bool cout_term = isatty(fileno(stdout)) != 0;
    return cout_term;
  } else if (osbuf == std::cerr.rdbuf() || osbuf == std::clog.rdbuf()) {
    static const bool cerr_term = isatty(fileno(stderr)) != 0;
    return cerr_term;
  }
  return false;
}

}  // namespace tachyon::base
