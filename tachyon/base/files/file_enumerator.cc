// Copyright 2013 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/files/file_enumerator.h"

#include "tachyon/base/files/file_util.h"
#include "tachyon/base/functional/function_ref.h"

namespace tachyon::base {

FileEnumerator::FileInfo::~FileInfo() = default;

bool FileEnumerator::ShouldSkip(const FilePath& path) {
  std::string basename = path.BaseName().value();
  return basename == "." ||
         (basename == ".." &&
          !(INCLUDE_DOT_DOT & file_type_));
}

bool FileEnumerator::IsTypeMatched(bool is_dir) const {
  return (file_type_ &
          (is_dir ? FileEnumerator::DIRECTORIES : FileEnumerator::FILES)) != 0;
}

void FileEnumerator::ForEach(FunctionRef<void(const FilePath& path)> ref) {
  for (FilePath name = Next(); !name.empty(); name = Next()) {
    ref(name);
  }
}

}  // namespace tachyon::base
