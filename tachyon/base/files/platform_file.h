// Copyright 2017 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_FILES_PLATFORM_FILE_H_
#define TACHYON_BASE_FILES_PLATFORM_FILE_H_

#include "tachyon/base/files/scoped_file.h"
#include "tachyon/build/build_config.h"

#if BUILDFLAG(IS_WIN)
#include "tachyon/base/win/scoped_handle.h"
#include "tachyon/base/win/windows_types.h"
#endif

// This file defines platform-independent types for dealing with
// platform-dependent files. If possible, use the higher-level base::File class
// rather than these primitives.

namespace tachyon::base {

#if BUILDFLAG(IS_WIN)

using PlatformFile = HANDLE;
using ScopedPlatformFile = ::tachyon::base::win::ScopedHandle;

// It would be nice to make this constexpr but INVALID_HANDLE_VALUE is a
// ((void*)(-1)) which Clang rejects since reinterpret_cast is technically
// disallowed in constexpr. Visual Studio accepts this, however.
const PlatformFile kInvalidPlatformFile = INVALID_HANDLE_VALUE;

#elif BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)

using PlatformFile = int;
using ScopedPlatformFile = ::tachyon::base::ScopedFD;

constexpr PlatformFile kInvalidPlatformFile = -1;

#endif

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FILES_PLATFORM_FILE_H_
