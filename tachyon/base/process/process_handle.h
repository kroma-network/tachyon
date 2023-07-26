// Copyright 2013 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_PROCESS_PROCESS_HANDLE_H_
#define TACHYON_BASE_PROCESS_PROCESS_HANDLE_H_

#include <stdint.h>
#include <sys/types.h>

#include <iosfwd>

#include "tachyon/export.h"
#include "tachyon/build/build_config.h"

#if BUILDFLAG(IS_WIN)
#include "tachyon/base/win/windows_types.h"
#endif

#if BUILDFLAG(IS_FUCHSIA)
#include <zircon/types.h>
#endif

namespace tachyon {
namespace base {

class FilePath;

// ProcessHandle is a platform specific type which represents the underlying OS
// handle to a process.
// ProcessId is a number which identifies the process in the OS.
#if BUILDFLAG(IS_WIN)
typedef HANDLE ProcessHandle;
typedef DWORD ProcessId;
typedef HANDLE UserTokenHandle;
const ProcessHandle kNullProcessHandle = NULL;
const ProcessId kNullProcessId = 0;
#elif BUILDFLAG(IS_FUCHSIA)
typedef zx_handle_t ProcessHandle;
typedef zx_koid_t ProcessId;
const ProcessHandle kNullProcessHandle = ZX_HANDLE_INVALID;
const ProcessId kNullProcessId = ZX_KOID_INVALID;
#elif BUILDFLAG(IS_POSIX)
// On POSIX, our ProcessHandle will just be the PID.
typedef pid_t ProcessHandle;
typedef pid_t ProcessId;
const ProcessHandle kNullProcessHandle = 0;
const ProcessId kNullProcessId = 0;
#endif  // BUILDFLAG(IS_WIN)

// To print ProcessIds portably use CrPRIdPid (based on PRIuS and friends from
// C99 and format_macros.h) like this:
// base::StringPrintf("PID is %" CrPRIdPid ".\n", pid);
#if BUILDFLAG(IS_WIN) || BUILDFLAG(IS_FUCHSIA)
#define CrPRIdPid "ld"
#else
#define CrPRIdPid "d"
#endif

class UniqueProcId {
 public:
  explicit UniqueProcId(ProcessId value) : value_(value) {}
  UniqueProcId(const UniqueProcId& other) = default;
  UniqueProcId& operator=(const UniqueProcId& other) = default;

  // Returns the process PID. WARNING: On some platforms, the pid may not be
  // valid within the current process sandbox.
  ProcessId GetUnsafeValue() const { return value_; }

  bool operator==(const UniqueProcId& other) const {
    return value_ == other.value_;
  }

  bool operator!=(const UniqueProcId& other) const {
    return value_ != other.value_;
  }

  bool operator<(const UniqueProcId& other) const {
    return value_ < other.value_;
  }

  bool operator<=(const UniqueProcId& other) const {
    return value_ <= other.value_;
  }

  bool operator>(const UniqueProcId& other) const {
    return value_ > other.value_;
  }

  bool operator>=(const UniqueProcId& other) const {
    return value_ >= other.value_;
  }

 private:
  ProcessId value_;
};

std::ostream& operator<<(std::ostream& os, const UniqueProcId& obj);

// Returns the id of the current process.
// Note that on some platforms, this is not guaranteed to be unique across
// processes (use GetUniqueIdForProcess if uniqueness is required).
TACHYON_EXPORT ProcessId GetCurrentProcId();

// Returns a unique ID for the current process. The ID will be unique across all
// currently running processes within the chrome session, but IDs of terminated
// processes may be reused.
TACHYON_EXPORT UniqueProcId GetUniqueIdForProcess();

#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS)
// When a process is started in a different PID namespace from the browser
// process, this function must be called with the process's PID in the browser's
// PID namespace in order to initialize its unique ID. Not thread safe.
// WARNING: To avoid inconsistent results from GetUniqueIdForProcess, this
// should only be called very early after process startup - ideally as soon
// after process creation as possible.
TACHYON_EXPORT void InitUniqueIdForProcessInPidNamespace(
    ProcessId pid_outside_of_namespace);
#endif

// Returns the ProcessHandle of the current process.
TACHYON_EXPORT ProcessHandle GetCurrentProcessHandle();

// Returns the process ID for the specified process. This is functionally the
// same as Windows' GetProcessId(), but works on versions of Windows before Win
// XP SP1 as well.
// DEPRECATED. New code should be using Process::Pid() instead.
// Note that on some platforms, this is not guaranteed to be unique across
// processes.
TACHYON_EXPORT ProcessId GetProcId(ProcessHandle process);

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_BASE_PROCESS_PROCESS_HANDLE_H_
