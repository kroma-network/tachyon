// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/files/file_util.h"

#import <Foundation/Foundation.h>
#include <copyfile.h>
#include <stdlib.h>
#include <string.h>

#include "tachyon/base/logging.h"
#include "tachyon/base/files/file_path.h"
#include "tachyon/base/mac/foundation_util.h"
#include "tachyon/base/strings/string_util.h"
// #include "tachyon/base/threading/scoped_blocking_call.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

namespace tachyon::base {

bool CopyFile(const FilePath& from_path, const FilePath& to_path) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  if (from_path.ReferencesParent() || to_path.ReferencesParent())
    return false;
  return (copyfile(from_path.value().c_str(), to_path.value().c_str(),
                   /*state=*/nullptr, COPYFILE_DATA) == 0);
}

bool GetTempDir(base::FilePath* path) {
  // In order to facilitate hermetic runs on macOS, first check
  // MAC_CHROMIUM_TMPDIR. This is used instead of TMPDIR for historical reasons.
  // This was originally done for https://crbug.com/698759 (TMPDIR too long for
  // process singleton socket path), but is hopefully obsolete as of
  // https://crbug.com/1266817 (allows a longer process singleton socket path).
  // Continue tracking MAC_CHROMIUM_TMPDIR as that's what build infrastructure
  // sets on macOS.
  // const char* env_tmpdir = getenv("MAC_CHROMIUM_TMPDIR");
  const char* env_tmpdir = getenv("MAC_TACHYON_TMPDIR");
  if (env_tmpdir) {
    *path = base::FilePath(env_tmpdir);
    return true;
  }

  // If we didn't find it, fall back to the native function.
  NSString* tmp = NSTemporaryDirectory();
  if (tmp == nil)
    return false;
  *path = base::mac::NSStringToFilePath(tmp);
  return true;
}

FilePath GetHomeDir() {
  NSString* tmp = NSHomeDirectory();
  if (tmp != nil) {
    FilePath mac_home_dir = base::mac::NSStringToFilePath(tmp);
    if (!mac_home_dir.empty())
      return mac_home_dir;
  }

  // Fall back on temp dir if no home directory is defined.
  FilePath rv;
  if (GetTempDir(&rv))
    return rv;

  // Last resort.
  return FilePath("/tmp");
}

}  // namespace tachyon::base
