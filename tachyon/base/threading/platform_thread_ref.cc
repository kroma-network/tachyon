// Copyright 2021 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/threading/platform_thread_ref.h"

#include <ostream>

namespace tachyon::base {

std::ostream& operator<<(std::ostream& os, const PlatformThreadRef& ref) {
  os << ref.id_;
  return os;
}

}  // namespace tachyon::base
