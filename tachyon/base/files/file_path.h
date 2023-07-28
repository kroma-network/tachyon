// Copyright (c) 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_FILES_FILE_PATH_H_
#define TACHYON_BASE_FILES_FILE_PATH_H_

#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "tachyon/export.h"

namespace tachyon::base {

// This is a class represents file path inside tachyon projects.
// CAUTION: On linux, unlike your expectation, it's not utf-8 encoded, encoding
// is dependent to locale.
// NOTE: please do not use std::string directly!
class TACHYON_EXPORT FilePath {
 public:
  FilePath();
  explicit FilePath(std::string_view path);

  FilePath(const FilePath& other);
  FilePath& operator=(const FilePath& other);

  FilePath(FilePath&& other) noexcept;
  FilePath& operator=(FilePath&& other) noexcept;

  ~FilePath();

  bool operator==(const FilePath& other) const { return path_ == other.path_; }
  bool operator!=(const FilePath& other) const { return path_ != other.path_; }

  bool operator<(const FilePath& other) const { return path_ < other.path_; }

  const std::string& value() const { return path_; }

  bool empty() const { return path_.empty(); }

  void clear() { path_.clear(); }

  // Returns true if |path_| references to root. It also returns true |path_|
  // consists of '/';
  bool IsRoot() const;

  // Fills |components| with all of the components of |path_|. It is
  // equivalent to calling DirName().value() on the path's root component,
  // and BaseName().value() on each child component.
  //
  // To make sure this is lossless so we can differentiate absolute and
  // relative paths, the root slash will be included even though no other
  // slashes will be. So for "/foo/bar", |components| will be [ "/", "foo",
  // "bar" ]
  void GetComponents(std::vector<std::string>* components) const;

  // Returns true if this FilePath is a parent or ancestor of the |child|.
  // Absolute and relative paths are accepted i.e. /foo is a parent to /foo/bar,
  // and foo is a parent to foo/bar. Any ancestor is considered a parent i.e. /a
  // is a parent to both /a/b and /a/b/c.  Does not convert paths to absolute,
  // follow symlinks or directory navigation (e.g. ".."). A path is *NOT* its
  // own parent.
  bool IsParent(const FilePath& child) const;

  // If IsParent(child) holds, appends to path (if non-NULL) the
  // relative path to child and returns true.  For example, if parent
  // holds "/Users/johndoe/Library/Application Support", child holds
  // "/Users/johndoe/Library/Application Support/Google/Chrome/Default", and
  // *path holds "/Users/johndoe/Library/Caches", then after
  // parent.AppendRelativePath(child, path) is called *path will hold
  // "/Users/johndoe/Library/Caches/Google/Chrome/Default".  Otherwise,
  // returns false.
  bool AppendRelativePath(const FilePath& child, FilePath* path) const;

  // Returns the directory name of this FilePath. If |path_| is root, it
  // returns the same. For example, if |path_| is "foo/bar", it returns "foo".
  [[nodiscard]] FilePath DirName() const;

  // Returns the base name of this FilePath. If |path_| is root, it returns
  // empty path. For example, if |path_| is "foo/bar", it returns "bar".
  [[nodiscard]] FilePath BaseName() const;

  // Returns the extension of this FilePath. If |path_| doesn't contain an
  // extension, it returns empty string. For example, if |path_| is "foo.jpg",
  // it returns ".jpg".
  [[nodiscard]] std::string Extension() const;

  // Returns a FilePath by appending a separator and the supplied path
  // component to this object's path. Append takes care to avoid adding
  // excessive separators if this object's path already ends with a separator.
  //
  // NOTE: |component| must be a relative path, it is an error to pass an
  // absolute path.
  [[nodiscard]] FilePath Append(std::string_view component) const;
  [[nodiscard]] FilePath Append(const FilePath& component) const;

  // Convenient operators, which does the same above.
  FilePath operator/(std::string_view component) const;
  FilePath operator/(const FilePath& component) const;
  FilePath& operator/=(std::string_view component);
  FilePath& operator/=(const FilePath& component);

  // Returns true if |path_| is absolute.
  bool IsAbsolute() const;

  // Returns true if |path_| ends with a path separator character.
  [[nodiscard]] bool EndsWithSeparator() const;

  // Returns a copy of this FilePath that ends with a trailing separator. If
  // the input path is empty, an empty FilePath will be returned.
  [[nodiscard]] FilePath AsEndingWithSeparator() const;

  // Returns a copy of this FilePath that does not end with a trailing
  // separator.
  [[nodiscard]] FilePath StripTrailingSeparators() const;

  // Returns true if this FilePath contains an attempt to reference a parent
  // directory (e.g. has a path component that is "..").
  bool ReferencesParent() const;

 private:
  // Remove trailing separators from this object. If |path_| is absolute, it
  // will never be stripped any more than to refer to the absolute root
  // directory, so "////" will become "/", not "".
  void StripTrailingSeparatorsInternal();

  std::string path_;
};

TACHYON_EXPORT std::ostream& operator<<(std::ostream& out,
                                        const FilePath& file_path);

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FILES_FILE_PATH_H_
