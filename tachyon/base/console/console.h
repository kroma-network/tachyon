// Copyright (c) 2020 The Console Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_CONSOLE_CONSOLE_H_
#define TACHYON_BASE_CONSOLE_CONSOLE_H_

#include <ostream>

#include "tachyon/export.h"

namespace tachyon {
namespace base {

class TACHYON_EXPORT Console {
 public:
  struct Info {
    bool support_ansi = false;
    bool support_8bit_color = false;
    bool support_truecolor = false;

    Info();

   private:
    void Init();
  };

  static Info& GetInfo();
  static bool IsConnected(std::ostream& os);
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_BASE_CONSOLE_CONSOLE_H_
