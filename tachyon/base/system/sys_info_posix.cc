// Copyright 2011 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <unistd.h>

#include "tachyon/base/numerics/checked_math.h"
#include "tachyon/base/system/sys_info.h"

namespace tachyon::base {

// static
size_t SysInfo::VMAllocationGranularity() {
  return checked_cast<size_t>(getpagesize());
}

}  // namespace tachyon::base
