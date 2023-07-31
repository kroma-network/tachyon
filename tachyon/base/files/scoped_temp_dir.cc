// Copyright 2011 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/files/scoped_temp_dir.h"

#include "tachyon/base/files/file_util.h"
#include "tachyon/base/logging.h"

namespace tachyon::base {

namespace {

constexpr char kScopedDirPrefix[] = "scoped_dir";

}  // namespace

ScopedTempDir::ScopedTempDir() = default;

ScopedTempDir::ScopedTempDir(ScopedTempDir&& other) noexcept
    : path_(other.Take()) {}

ScopedTempDir& ScopedTempDir::operator=(ScopedTempDir&& other) {
  if (!path_.empty() && !Delete())
    DLOG(WARNING) << "Could not delete temp dir in operator=().";
  path_ = other.Take();
  return *this;
}

ScopedTempDir::~ScopedTempDir() {
  if (!path_.empty() && !Delete())
    DLOG(WARNING) << "Could not delete temp dir in dtor.";
}

bool ScopedTempDir::CreateUniqueTempDir() {
  if (!path_.empty())
    return false;

  // This "scoped_dir" prefix is only used on Windows and serves as a template
  // for the unique name.
  if (!CreateNewTempDirectory(kScopedDirPrefix, &path_))
    return false;

  return true;
}

bool ScopedTempDir::CreateUniqueTempDirUnderPath(const FilePath& base_path) {
  if (!path_.empty())
    return false;

  // If |base_path| does not exist, create it.
  if (!CreateDirectory(base_path))
    return false;

  // Create a new, uniquely named directory under |base_path|.
  if (!CreateTemporaryDirInDir(base_path, kScopedDirPrefix, &path_))
    return false;

  return true;
}

bool ScopedTempDir::Set(const FilePath& path) {
  if (!path_.empty())
    return false;

  if (!DirectoryExists(path) && !CreateDirectory(path))
    return false;

  path_ = path;
  return true;
}

bool ScopedTempDir::Delete() {
  if (path_.empty())
    return false;

  bool ret = DeletePathRecursively(path_);
  if (ret) {
    // We only clear the path if deleted the directory.
    path_.clear();
  }

  return ret;
}

FilePath ScopedTempDir::Take() {
  return std::exchange(path_, FilePath());
}

const FilePath& ScopedTempDir::GetPath() const {
  DCHECK(!path_.empty()) << "Did you call CreateUniqueTempDir* before?";
  return path_;
}

bool ScopedTempDir::IsValid() const {
  return !path_.empty() && DirectoryExists(path_);
}

// static
const char* ScopedTempDir::GetTempDirPrefix() {
  return kScopedDirPrefix;
}

}  // namespace tachyon::base
