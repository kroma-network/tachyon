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
  // Null-terminated array of separators used to separate components in paths.
  // Each character in this array is a valid separator, but kSeparators[0] is
  // treated as the canonical separator and is used when composing pathnames.
  static constexpr char kSeparators[] = "/";

  // std::size(kSeparators), i.e., the number of separators in kSeparators plus
  // one (the null terminator at the end of kSeparators).
  static constexpr size_t kSeparatorsLength = std::size(kSeparators);

  // The special path component meaning "this directory."
  static constexpr char kCurrentDirectory[] = ".";

  // The special path component meaning "the parent directory."
  static constexpr char kParentDirectory[] = "..";

  // The character used to identify a file extension.
  static constexpr char kExtensionSeparator = '.';

  // The special path to identify a root.
  static constexpr char kRootPath[] = "/";

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

  // Returns true if |character| is in kSeparators.
  static bool IsSeparator(char character);

  // Returns true if |path_| references to root. It also returns true |path_|
  // consists of '/';
  bool IsRoot() const;

  // Returns a vector of all of the components of the provided path. It is
  // equivalent to calling DirName().value() on the path's root component,
  // and BaseName().value() on each child component.
  //
  // To make sure this is lossless so we can differentiate absolute and
  // relative paths, the root slash will be included even though no other
  // slashes will be. The precise behavior is:
  //
  // Posix:  "/foo/bar"  ->  [ "/", "foo", "bar" ]
  // Windows:  "C:\foo\bar"  ->  [ "C:", "\\", "foo", "bar" ]
  std::vector<std::string> GetComponents() const;

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

  // Returns the final extension of a file path, or an empty string if the file
  // path has no extension.  In most cases, the final extension of a file path
  // refers to the part of the file path from the last dot to the end (including
  // the dot itself).  For example, this method applied to "/pics/jojo.jpg"
  // and "/pics/jojo." returns ".jpg" and ".", respectively.  However, if the
  // base name of the file path is either "." or "..", this method returns an
  // empty string.
  //
  // TODO(davidben): Check all our extension-sensitive code to see if
  // we can rename this to Extension() and the other to something like
  // LongExtension(), defaulting to short extensions and leaving the
  // long "extensions" to logic like base::GetUniquePathNumber().
  [[nodiscard]] std::string FinalExtension() const;

  // Returns "/pics/jojo" for path "/pics/jojo.jpg"
  // NOTE: this is slightly different from the similar file_util implementation
  // which returned simply 'jojo'.
  [[nodiscard]] FilePath RemoveExtension() const;

  // Removes the path's file extension, as in RemoveExtension(), but
  // ignores double extensions.
  [[nodiscard]] FilePath RemoveFinalExtension() const;

  // Inserts |suffix| after the file name portion of |path| but before the
  // extension.  Returns "" if BaseName() == "." or "..".
  // Examples:
  // path == "jojo.jpg"         suffix == " (1)", returns "jojo (1).jpg"
  [[nodiscard]] FilePath InsertBeforeExtension(std::string_view suffix) const;

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
