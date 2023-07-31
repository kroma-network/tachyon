// Copyright 2011 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_FILES_SCOPED_TEMP_DIR_H_
#define TACHYON_BASE_FILES_SCOPED_TEMP_DIR_H_

// An object representing a temporary / scratch directory that should be
// cleaned up (recursively) when this object goes out of scope.  Since deletion
// occurs during the destructor, no further error handling is possible if the
// directory fails to be deleted.  As a result, deletion is not guaranteed by
// this class.  (However note that, whenever possible, by default
// CreateUniqueTempDir creates the directory in a location that is
// automatically cleaned up on reboot, or at other appropriate times.)
//
// Multiple calls to the methods which establish a temporary directory
// (CreateUniqueTempDir, CreateUniqueTempDirUnderPath, and Set) must have
// intervening calls to Delete or Take, or the calls will fail.

#include "tachyon/export.h"
#include "tachyon/base/files/file_path.h"

namespace tachyon::base {

class TACHYON_EXPORT ScopedTempDir {
 public:
  // No directory is owned/created initially.
  ScopedTempDir();

  ScopedTempDir(ScopedTempDir&&) noexcept;
  ScopedTempDir& operator=(ScopedTempDir&&);

  // Recursively delete path.
  ~ScopedTempDir();

  // Creates a unique directory in TempPath, and takes ownership of it.
  // See file_util::CreateNewTemporaryDirectory.
  [[nodiscard]] bool CreateUniqueTempDir();

  // Creates a unique directory under a given path, and takes ownership of it.
  [[nodiscard]] bool CreateUniqueTempDirUnderPath(const FilePath& path);

  // Takes ownership of directory at |path|, creating it if necessary.
  // Don't call multiple times unless Take() has been called first.
  [[nodiscard]] bool Set(const FilePath& path);

  // Deletes the temporary directory wrapped by this object.
  [[nodiscard]] bool Delete();

  // Caller takes ownership of the temporary directory so it won't be destroyed
  // when this object goes out of scope.
  FilePath Take();

  // Returns the path to the created directory. Call one of the
  // CreateUniqueTempDir* methods before getting the path.
  const FilePath& GetPath() const;

  // Returns true if path_ is non-empty and exists.
  bool IsValid() const;

  // Returns the prefix used for temp directory names generated by
  // ScopedTempDirs.
  static const char* GetTempDirPrefix();

 private:
  FilePath path_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FILES_SCOPED_TEMP_DIR_H_
