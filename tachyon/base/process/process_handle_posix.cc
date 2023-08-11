// Copyright 2013 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/process/process_handle.h"

#include <unistd.h>

namespace tachyon::base {

ProcessId GetCurrentProcId() {
  return getpid();
}

ProcessHandle GetCurrentProcessHandle() {
  return GetCurrentProcId();
}

ProcessId GetProcId(ProcessHandle process) {
  return process;
}

}  // namespace tachyon::base
