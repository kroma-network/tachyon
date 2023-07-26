// Copyright 2015 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/process/process_handle.h"

#include <stdint.h>

#include <ostream>

#include "tachyon/base/logging.h"
#include "tachyon/build/build_config.h"

namespace tachyon {
namespace base {

namespace {
ProcessId g_pid_outside_of_namespace = kNullProcessId;
}  // namespace

std::ostream& operator<<(std::ostream& os, const UniqueProcId& obj) {
  os << obj.GetUnsafeValue();
  return os;
}

UniqueProcId GetUniqueIdForProcess() {
  // Used for logging. Must not use LogMessage or any of the macros that call
  // into it.
  return (g_pid_outside_of_namespace != kNullProcessId)
             ? UniqueProcId(g_pid_outside_of_namespace)
             : UniqueProcId(GetCurrentProcId());
}

#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_AIX)

void InitUniqueIdForProcessInPidNamespace(ProcessId pid_outside_of_namespace) {
  DCHECK(pid_outside_of_namespace != kNullProcessId);
  g_pid_outside_of_namespace = pid_outside_of_namespace;
}

#endif

}  // namespace base
}  // namespace tachyon
