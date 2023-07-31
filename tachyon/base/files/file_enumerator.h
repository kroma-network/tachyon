// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_FILES_FILE_ENUMERATOR_H_
#define TACHYON_BASE_FILES_FILE_ENUMERATOR_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "tachyon/base/functional/function_ref.h"

#include "tachyon/export.h"
#include "tachyon/base/containers/stack.h"
#include "tachyon/base/files/file.h"
#include "tachyon/base/files/file_path.h"
#include "tachyon/base/time/time.h"
#include "tachyon/build/build_config.h"

#if BUILDFLAG(IS_WIN)
#include "tachyon/base/win/windows_types.h"
#elif BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#endif

namespace tachyon::base {

// A class for enumerating the files in a provided path. The order of the
// results is not guaranteed.
//
// This is blocking. Do not use on critical threads.
//
// Example:
//
//   base::FileEnumerator e(my_dir, false, base::FileEnumerator::FILES,
//                          "*.txt");
// Using `ForEach` with a lambda:
//   e.ForEach([](const base::FilePath& item) {...});
// Using a `for` loop:
//   for (base::FilePath name = e.Next(); !name.empty(); name = e.Next())
//     ...
class TACHYON_EXPORT FileEnumerator {
 public:
  // Note: copy & assign supported.
  class TACHYON_EXPORT FileInfo {
   public:
    FileInfo();
    ~FileInfo();

    bool IsDirectory() const;

    // The name of the file. This will not include any path information. This
    // is in constrast to the value returned by FileEnumerator.Next() which
    // includes the |root_path| passed into the FileEnumerator constructor.
    FilePath GetName() const;

    int64_t GetSize() const;

    // On POSIX systems, this is rounded down to the second.
    Time GetLastModifiedTime() const;

#if BUILDFLAG(IS_WIN)
    // Note that the cAlternateFileName (used to hold the "short" 8.3 name)
    // of the WIN32_FIND_DATA will be empty. Since we don't use short file
    // names, we tell Windows to omit it which speeds up the query slightly.
    const WIN32_FIND_DATA& find_data() const {
      return *ChromeToWindowsType(&find_data_);
    }
#elif BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
    const stat_wrapper_t& stat() const { return stat_; }
#endif

   private:
    friend class FileEnumerator;

#if BUILDFLAG(IS_WIN)
    CHROME_WIN32_FIND_DATA find_data_;
#elif BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
    stat_wrapper_t stat_;
    FilePath filename_;
#endif
  };

  enum FileType {
    FILES = 1 << 0,
    DIRECTORIES = 1 << 1,
    INCLUDE_DOT_DOT = 1 << 2,

    // Report only the names of entries and not their type, size, or
    // last-modified time. May only be used for non-recursive enumerations, and
    // implicitly includes both files and directories (neither of which may be
    // specified). When used, an enumerator's `GetInfo()` method must not be
    // called.
    NAMES_ONLY = 1 << 3,

#if BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
    SHOW_SYM_LINKS = 1 << 4,
#endif
  };

  // Search policy for intermediate folders.
  enum class FolderSearchPolicy {
    // Recursive search will pass through folders whose names match the
    // pattern. Inside each one, all files will be returned. Folders with names
    // that do not match the pattern will be ignored within their interior.
    MATCH_ONLY,
    // Recursive search will pass through every folder and perform pattern
    // matching inside each one.
    ALL,
  };

  // Determines how a FileEnumerator handles errors encountered during
  // enumeration. When no ErrorPolicy is explicitly set, FileEnumerator defaults
  // to IGNORE_ERRORS.
  enum class ErrorPolicy {
    // Errors are ignored if possible and FileEnumerator returns as many files
    // as it is able to enumerate.
    IGNORE_ERRORS,

    // Any error encountered during enumeration will terminate the enumeration
    // immediately. An error code indicating the nature of a failure can be
    // retrieved from |GetError()|.
    STOP_ENUMERATION,
  };

  // |root_path| is the starting directory to search for. It may or may not end
  // in a slash.
  //
  // If |recursive| is true, this will enumerate all matches in any
  // subdirectories matched as well. It does a breadth-first search, so all
  // files in one directory will be returned before any files in a
  // subdirectory.
  //
  // |file_type|, a bit mask of FileType, specifies whether the enumerator
  // should match files, directories, or both.
  //
  // |pattern| is an optional pattern for which files to match. This
  // works like shell globbing. For example, "*.txt" or "Foo???.doc".
  // However, be careful in specifying patterns that aren't cross platform
  // since the underlying code uses OS-specific matching routines.  In general,
  // Windows matching is less featureful than others, so test there first.
  // If unspecified, this will match all files.
  //
  // |folder_search_policy| optionally specifies a search behavior. Refer to
  // |FolderSearchPolicy| for a list of folder search policies and the meaning
  // of them. If |recursive| is false, this parameter has no effect.
  //
  // |error_policy| optionally specifies the behavior when an error occurs.
  // Refer to |ErrorPolicy| for a list of error policies and the meaning of
  // them.
  FileEnumerator(const FilePath& root_path, bool recursive, int file_type);
  FileEnumerator(const FilePath& root_path,
                 bool recursive,
                 int file_type,
                 const std::string& pattern);
  FileEnumerator(const FilePath& root_path,
                 bool recursive,
                 int file_type,
                 const std::string& pattern,
                 FolderSearchPolicy folder_search_policy);
  FileEnumerator(const FilePath& root_path,
                 bool recursive,
                 int file_type,
                 const std::string& pattern,
                 FolderSearchPolicy folder_search_policy,
                 ErrorPolicy error_policy);
  FileEnumerator(const FileEnumerator&) = delete;
  FileEnumerator& operator=(const FileEnumerator&) = delete;
  ~FileEnumerator();

  // Calls `ref` synchronously for each path found by the `FileEnumerator`. Each
  // path will incorporate the `root_path` passed in the constructor:
  // "<root_path>/file_name.txt". If the `root_path` is absolute, then so will
  // be the paths provided in the `ref` invocations.
  void ForEach(FunctionRef<void(const FilePath& path)> ref);

  // Returns the next file or an empty string if there are no more results.
  //
  // The returned path will incorporate the |root_path| passed in the
  // constructor: "<root_path>/file_name.txt". If the |root_path| is absolute,
  // then so will be the result of Next().
  FilePath Next();

  // Returns info about the file last returned by Next(). Note that on Windows
  // and Fuchsia, GetInfo() does not play well with INCLUDE_DOT_DOT. In
  // particular, the GetLastModifiedTime() for the .. directory is 1601-01-01
  // on Fuchsia (https://crbug.com/1106172) and is equal to the last modified
  // time of the current directory on Windows (https://crbug.com/1119546).
  // Must not be used with FileType::NAMES_ONLY.
  FileInfo GetInfo() const;

  // Once |Next()| returns an empty path, enumeration has been terminated. If
  // termination was normal (i.e. no more results to enumerate) or ErrorPolicy
  // is set to IGNORE_ERRORS, this returns FILE_OK. Otherwise it returns an
  // error code reflecting why enumeration was stopped early.
  File::Error GetError() const { return error_; }

 private:
  // Returns true if the given path should be skipped in enumeration.
  bool ShouldSkip(const FilePath& path);

  bool IsTypeMatched(bool is_dir) const;

  bool IsPatternMatched(const FilePath& src) const;

#if BUILDFLAG(IS_WIN)
  const WIN32_FIND_DATA& find_data() const {
    return *ChromeToWindowsType(&find_data_);
  }

  // True when find_data_ is valid.
  bool has_find_data_ = false;
  CHROME_WIN32_FIND_DATA find_data_;
  HANDLE find_handle_ = INVALID_HANDLE_VALUE;
#elif BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
  // The files in the current directory
  std::vector<FileInfo> directory_entries_;

  // Set of visited directories. Used to prevent infinite looping along
  // circular symlinks.
  // The Android NDK (r23) does not declare `st_ino` as an `ino_t`, hence the
  // need for the ugly decltype.
  std::unordered_set<decltype(stat_wrapper_t::st_ino)> visited_directories_;

  // The next entry to use from the directory_entries_ vector
  size_t current_directory_entry_;
#endif
  FilePath root_path_;
  const bool recursive_;
  int file_type_;
  std::string pattern_;
  const FolderSearchPolicy folder_search_policy_;
  const ErrorPolicy error_policy_;
  File::Error error_ = File::FILE_OK;

  // A stack that keeps track of which subdirectories we still need to
  // enumerate in the breadth-first search.
  base::stack<FilePath> pending_paths_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FILES_FILE_ENUMERATOR_H_
