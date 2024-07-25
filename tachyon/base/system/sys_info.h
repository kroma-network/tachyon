// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_SYSTEM_SYS_INFO_H_
#define TACHYON_BASE_SYSTEM_SYS_INFO_H_

#include <stddef.h>

#include "tachyon/export.h"

namespace tachyon::base {

class TACHYON_EXPORT SysInfo {
 public:
  // Return the smallest amount of memory (in bytes) which the VM system will
  // allocate.
  static size_t VMAllocationGranularity();
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_SYSTEM_SYS_INFO_H_
