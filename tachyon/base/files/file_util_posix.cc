// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.


#include "tachyon/base/files/file_util.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "absl/strings/str_cat.h"

#include "tachyon/export.h"
#include "tachyon/base/bits.h"
#include "tachyon/base/containers/adapters.h"
#include "tachyon/base/containers/contains.h"
#include "tachyon/base/containers/stack.h"
#include "tachyon/base/environment.h"
#include "tachyon/base/files/file_enumerator.h"
#include "tachyon/base/files/file_path.h"
#include "tachyon/base/files/scoped_file.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/no_destructor.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/numerics/safe_conversions.h"
// #include "tachyon/base/path_service.h"
#include "tachyon/base/posix/eintr_wrapper.h"
#include "tachyon/base/strings/string_util.h"
// #include "tachyon/base/strings/sys_string_conversions.h"
// #include "tachyon/base/strings/utf_string_conversions.h"
// #include "tachyon/base/system/sys_info.h"
// #include "tachyon/base/threading/scoped_blocking_call.h"
#include "tachyon/base/time/time.h"
#include "tachyon/build/build_config.h"

#if BUILDFLAG(IS_APPLE)
#include <AvailabilityMacros.h>
#include "tachyon/base/mac/foundation_util.h"
#endif

#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_ANDROID)
#include <sys/sendfile.h>
#endif

#if BUILDFLAG(IS_ANDROID)
#include "tachyon/base/android/content_uri_utils.h"
#include "tachyon/base/os_compat_android.h"
#endif

#if !BUILDFLAG(IS_IOS)
#include <grp.h>
#endif

// We need to do this on AIX due to some inconsistencies in how AIX
// handles XOPEN_SOURCE and ALL_SOURCE.
#if BUILDFLAG(IS_AIX)
extern "C" char* mkdtemp(char* path);
#endif

namespace tachyon::base {

namespace {

#if BUILDFLAG(IS_MAC)
// Helper for VerifyPathControlledByUser.
bool VerifySpecificPathControlledByUser(const FilePath& path,
                                        uid_t owner_uid,
                                        const std::set<gid_t>& group_gids) {
  stat_wrapper_t stat_info;
  if (File::Lstat(path.value().c_str(), &stat_info) != 0) {
    DPLOG(ERROR) << "Failed to get information on path "
                 << path.value();
    return false;
  }

  if (S_ISLNK(stat_info.st_mode)) {
    DLOG(ERROR) << "Path " << path.value() << " is a symbolic link.";
    return false;
  }

  if (stat_info.st_uid != owner_uid) {
    DLOG(ERROR) << "Path " << path.value() << " is owned by the wrong user.";
    return false;
  }

  if ((stat_info.st_mode & S_IWGRP) &&
      !Contains(group_gids, stat_info.st_gid)) {
    DLOG(ERROR) << "Path " << path.value()
                << " is writable by an unprivileged group.";
    return false;
  }

  if (stat_info.st_mode & S_IWOTH) {
    DLOG(ERROR) << "Path " << path.value() << " is writable by any user.";
    return false;
  }

  return true;
}
#endif

base::FilePath GetTempTemplate() {
  return FormatTemporaryFileName("XXXXXX");
}

bool AdvanceEnumeratorWithStat(FileEnumerator* traversal,
                               FilePath* out_next_path,
                               stat_wrapper_t* out_next_stat) {
  DCHECK(out_next_path);
  DCHECK(out_next_stat);
  *out_next_path = traversal->Next();
  if (out_next_path->empty())
    return false;

  *out_next_stat = traversal->GetInfo().stat();
  return true;
}

bool DoCopyDirectory(const FilePath& from_path,
                     const FilePath& to_path,
                     bool recursive,
                     bool open_exclusive) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  // Some old callers of CopyDirectory want it to support wildcards.
  // After some discussion, we decided to fix those callers.
  // Break loudly here if anyone tries to do this.
  DCHECK(to_path.value().find('*') == std::string::npos);
  DCHECK(from_path.value().find('*') == std::string::npos);

  if (from_path.value().size() >= PATH_MAX) {
    return false;
  }

  // This function does not properly handle destinations within the source
  FilePath real_to_path = to_path;
  if (PathExists(real_to_path))
    real_to_path = MakeAbsoluteFilePath(real_to_path);
  else
    real_to_path = MakeAbsoluteFilePath(real_to_path.DirName());
  if (real_to_path.empty())
    return false;

  FilePath real_from_path = MakeAbsoluteFilePath(from_path);
  if (real_from_path.empty())
    return false;
  if (real_to_path == real_from_path || real_from_path.IsParent(real_to_path))
    return false;

  int traverse_type = FileEnumerator::FILES | FileEnumerator::SHOW_SYM_LINKS;
  if (recursive)
    traverse_type |= FileEnumerator::DIRECTORIES;
  FileEnumerator traversal(from_path, recursive, traverse_type);

  // We have to mimic windows behavior here. |to_path| may not exist yet,
  // start the loop with |to_path|.
  stat_wrapper_t from_stat;
  FilePath current = from_path;
  if (File::Stat(from_path.value().c_str(), &from_stat) < 0) {
    DPLOG(ERROR) << "CopyDirectory() couldn't stat source directory: "
                 << from_path.value();
    return false;
  }
  FilePath from_path_base = from_path;
  if (recursive && DirectoryExists(to_path)) {
    // If the destination already exists and is a directory, then the
    // top level of source needs to be copied.
    from_path_base = from_path.DirName();
  }

  // The Windows version of this function assumes that non-recursive calls
  // will always have a directory for from_path.
  // TODO(maruel): This is not necessary anymore.
  DCHECK(recursive || S_ISDIR(from_stat.st_mode));

  do {
    // current is the source path, including from_path, so append
    // the suffix after from_path to to_path to create the target_path.
    FilePath target_path(to_path);
    if (from_path_base != current &&
        !from_path_base.AppendRelativePath(current, &target_path)) {
      return false;
    }

    if (S_ISDIR(from_stat.st_mode)) {
      mode_t mode = (from_stat.st_mode & 01777) | S_IRUSR | S_IXUSR | S_IWUSR;
      if (mkdir(target_path.value().c_str(), mode) == 0)
        continue;
      if (errno == EEXIST && !open_exclusive)
        continue;

      DPLOG(ERROR) << "CopyDirectory() couldn't create directory: "
                   << target_path.value();
      return false;
    }

    if (!S_ISREG(from_stat.st_mode)) {
      DLOG(WARNING) << "CopyDirectory() skipping non-regular file: "
                    << current.value();
      continue;
    }

    // Add O_NONBLOCK so we can't block opening a pipe.
    File infile(open(current.value().c_str(), O_RDONLY | O_NONBLOCK));
    if (!infile.IsValid()) {
      DPLOG(ERROR) << "CopyDirectory() couldn't open file: " << current.value();
      return false;
    }

    stat_wrapper_t stat_at_use;
    if (File::Fstat(infile.GetPlatformFile(), &stat_at_use) < 0) {
      DPLOG(ERROR) << "CopyDirectory() couldn't stat file: " << current.value();
      return false;
    }

    if (!S_ISREG(stat_at_use.st_mode)) {
      DLOG(WARNING) << "CopyDirectory() skipping non-regular file: "
                    << current.value();
      continue;
    }

    int open_flags = O_WRONLY | O_CREAT;
    // If |open_exclusive| is set then we should always create the destination
    // file, so O_NONBLOCK is not necessary to ensure we don't block on the
    // open call for the target file below, and since the destination will
    // always be a regular file it wouldn't affect the behavior of the
    // subsequent write calls anyway.
    if (open_exclusive)
      open_flags |= O_EXCL;
    else
      open_flags |= O_TRUNC | O_NONBLOCK;
    // Each platform has different default file opening modes for CopyFile which
    // we want to replicate here. On OS X, we use copyfile(3) which takes the
    // source file's permissions into account. On the other platforms, we just
    // use the base::File constructor. On Chrome OS, base::File uses a different
    // set of permissions than it does on other POSIX platforms.
#if BUILDFLAG(IS_APPLE)
    mode_t mode = 0600 | (stat_at_use.st_mode & 0177);
#elif BUILDFLAG(IS_CHROMEOS)
    mode_t mode = 0644;
#else
    mode_t mode = 0600;
#endif
    File outfile(open(target_path.value().c_str(), open_flags, mode));
    if (!outfile.IsValid()) {
      DPLOG(ERROR) << "CopyDirectory() couldn't create file: "
                   << target_path.value();
      return false;
    }

    if (!CopyFileContents(infile, outfile)) {
      DLOG(ERROR) << "CopyDirectory() couldn't copy file: " << current.value();
      return false;
    }
  } while (AdvanceEnumeratorWithStat(&traversal, &current, &from_stat));

  return true;
}

// TODO(erikkay): The Windows version of this accepts paths like "foo/bar/*"
// which works both with and without the recursive flag.  I'm not sure we need
// that functionality. If not, remove from file_util_win.cc, otherwise add it
// here.
bool DoDeleteFile(const FilePath& path, bool recursive) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);

#if BUILDFLAG(IS_ANDROID)
  if (path.IsContentUri())
    return DeleteContentUri(path);
#endif  // BUILDFLAG(IS_ANDROID)

  const char* path_str = path.value().c_str();
  stat_wrapper_t file_info;
  if (File::Lstat(path_str, &file_info) != 0) {
    // The Windows version defines this condition as success.
    return (errno == ENOENT);
  }
  if (!S_ISDIR(file_info.st_mode))
    return (unlink(path_str) == 0) || (errno == ENOENT);
  if (!recursive)
    return (rmdir(path_str) == 0) || (errno == ENOENT);

  bool success = true;
  stack<std::string> directories;
  directories.push(path.value());
  FileEnumerator traversal(path, true,
      FileEnumerator::FILES | FileEnumerator::DIRECTORIES |
      FileEnumerator::SHOW_SYM_LINKS);
  for (FilePath current = traversal.Next(); !current.empty();
       current = traversal.Next()) {
    if (traversal.GetInfo().IsDirectory())
      directories.push(current.value());
    else
      success &= (unlink(current.value().c_str()) == 0) || (errno == ENOENT);
  }

  while (!directories.empty()) {
    FilePath dir = FilePath(directories.top());
    directories.pop();
    success &= (rmdir(dir.value().c_str()) == 0) || (errno == ENOENT);
  }
  return success;
}

#if !BUILDFLAG(IS_APPLE)
// Appends |mode_char| to |mode| before the optional character set encoding; see
// https://www.gnu.org/software/libc/manual/html_node/Opening-Streams.html for
// details.
std::string AppendModeCharacter(std::string_view mode, char mode_char) {
  std::string result(mode);
  size_t comma_pos = result.find(',');
  result.insert(comma_pos == std::string::npos ? result.length() : comma_pos, 1,
                mode_char);
  return result;
}
#endif

}  // namespace

FilePath MakeAbsoluteFilePath(const FilePath& input) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  char full_path[PATH_MAX];
  if (realpath(input.value().c_str(), full_path) == nullptr)
    return FilePath();
  return FilePath(full_path);
}

std::optional<FilePath> MakeAbsoluteFilePathNoResolveSymbolicLinks(
    const FilePath& input) {
  if (input.empty()) {
    return std::nullopt;
  }

  FilePath collapsed_path;
  std::vector<std::string> components = input.GetComponents();
  absl::Span<std::string> components_span(components);
  // Start with root for absolute |input| and the current working directory for
  // a relative |input|.
  if (input.IsAbsolute()) {
    collapsed_path = FilePath(components_span[0]);
    components_span = components_span.subspan(1);
  } else {
    if (!GetCurrentDirectory(&collapsed_path)) {
      return std::nullopt;
    }
  }

  for (const auto& component : components_span) {
    if (component == FilePath::kCurrentDirectory) {
      continue;
    }

    if (component == FilePath::kParentDirectory) {
      // Pop the most recent component off the FilePath. Works correctly when
      // the FilePath is root.
      collapsed_path = collapsed_path.DirName();
      continue;
    }

    // This is just a regular component. Append it.
    collapsed_path = collapsed_path.Append(component);
  }

  return collapsed_path;
}

bool DeleteFile(const FilePath& path) {
  return DoDeleteFile(path, /*recursive=*/false);
}

bool DeletePathRecursively(const FilePath& path) {
  return DoDeleteFile(path, /*recursive=*/true);
}

bool ReplaceFile(const FilePath& from_path,
                 const FilePath& to_path,
                 File::Error* error) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  if (rename(from_path.value().c_str(), to_path.value().c_str()) == 0)
    return true;
  if (error)
    *error = File::GetLastFileError();
  return false;
}

bool CopyDirectory(const FilePath& from_path,
                   const FilePath& to_path,
                   bool recursive) {
  return DoCopyDirectory(from_path, to_path, recursive, false);
}

bool CopyDirectoryExcl(const FilePath& from_path,
                       const FilePath& to_path,
                       bool recursive) {
  return DoCopyDirectory(from_path, to_path, recursive, true);
}

bool CreatePipe(ScopedFD* read_fd, ScopedFD* write_fd, bool non_blocking) {
  int fds[2];
  bool created =
      non_blocking ? CreateLocalNonBlockingPipe(fds) : (0 == pipe(fds));
  if (!created)
    return false;
  read_fd->reset(fds[0]);
  write_fd->reset(fds[1]);
  return true;
}

bool CreateLocalNonBlockingPipe(int fds[2]) {
#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS)
  return pipe2(fds, O_CLOEXEC | O_NONBLOCK) == 0;
#else
  int raw_fds[2];
  if (pipe(raw_fds) != 0)
    return false;
  ScopedFD fd_out(raw_fds[0]);
  ScopedFD fd_in(raw_fds[1]);
  if (!SetCloseOnExec(fd_out.get()))
    return false;
  if (!SetCloseOnExec(fd_in.get()))
    return false;
  if (!SetNonBlocking(fd_out.get()))
    return false;
  if (!SetNonBlocking(fd_in.get()))
    return false;
  fds[0] = fd_out.release();
  fds[1] = fd_in.release();
  return true;
#endif
}

bool SetNonBlocking(int fd) {
  const int flags = fcntl(fd, F_GETFL);
  if (flags == -1)
    return false;
  if (flags & O_NONBLOCK)
    return true;
  if (HANDLE_EINTR(fcntl(fd, F_SETFL, flags | O_NONBLOCK)) == -1)
    return false;
  return true;
}

bool SetCloseOnExec(int fd) {
  const int flags = fcntl(fd, F_GETFD);
  if (flags == -1)
    return false;
  if (flags & FD_CLOEXEC)
    return true;
  if (HANDLE_EINTR(fcntl(fd, F_SETFD, flags | FD_CLOEXEC)) == -1)
    return false;
  return true;
}

bool PathExists(const FilePath& path) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
#if BUILDFLAG(IS_ANDROID)
  if (path.IsContentUri()) {
    return ContentUriExists(path);
  }
#endif
  return access(path.value().c_str(), F_OK) == 0;
}

bool PathIsReadable(const FilePath& path) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  return access(path.value().c_str(), R_OK) == 0;
}

bool PathIsWritable(const FilePath& path) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  return access(path.value().c_str(), W_OK) == 0;
}

bool DirectoryExists(const FilePath& path) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  stat_wrapper_t file_info;
  if (File::Stat(path.value().c_str(), &file_info) != 0)
    return false;
  return S_ISDIR(file_info.st_mode);
}

bool ReadFromFD(int fd, char* buffer, size_t bytes) {
  size_t total_read = 0;
  while (total_read < bytes) {
    ssize_t bytes_read =
        HANDLE_EINTR(read(fd, buffer + total_read, bytes - total_read));
    if (bytes_read <= 0)
      break;
    total_read += static_cast<size_t>(bytes_read);
  }
  return total_read == bytes;
}

ScopedFD CreateAndOpenFdForTemporaryFileInDir(const FilePath& directory,
                                              FilePath* path) {
  /*
  TODO(chokobole):
  ScopedBlockingCall scoped_blocking_call(
      FROM_HERE,
      BlockingType::MAY_BLOCK);  // For call to mkstemp().
  */
  *path = directory.Append(GetTempTemplate());
  const std::string& tmpdir_string = path->value();
  // this should be OK since mkstemp just replaces characters in place
  char* buffer = const_cast<char*>(tmpdir_string.c_str());

  return ScopedFD(HANDLE_EINTR(mkstemp(buffer)));
}

#if !BUILDFLAG(IS_FUCHSIA)
bool CreateSymbolicLink(const FilePath& target_path,
                        const FilePath& symlink_path) {
  DCHECK(!symlink_path.empty());
  DCHECK(!target_path.empty());
  return ::symlink(target_path.value().c_str(),
                   symlink_path.value().c_str()) != -1;
}

bool ReadSymbolicLink(const FilePath& symlink_path, FilePath* target_path) {
  DCHECK(!symlink_path.empty());
  DCHECK(target_path);
  char buf[PATH_MAX];
  ssize_t count = ::readlink(symlink_path.value().c_str(), buf, std::size(buf));

#if BUILDFLAG(IS_ANDROID) && defined(__LP64__)
  // A few 64-bit Android L/M devices return INT_MAX instead of -1 here for
  // errors; this is related to bionic's (incorrect) definition of ssize_t as
  // being long int instead of int. Cast it so the compiler generates the
  // comparison we want here. https://crbug.com/1101940
  bool error = static_cast<int32_t>(count) <= 0;
#else
  bool error = count <= 0;
#endif

  if (error) {
    target_path->clear();
    return false;
  }

  *target_path =
      FilePath(std::string(buf, static_cast<size_t>(count)));
  return true;
}

std::optional<FilePath> ReadSymbolicLinkAbsolute(
    const FilePath& symlink_path) {
  FilePath target_path;
  if (!ReadSymbolicLink(symlink_path, &target_path)) {
    return std::nullopt;
  }

  // Relative symbolic links are relative to the symlink's directory.
  if (!target_path.IsAbsolute()) {
    target_path = symlink_path.DirName().Append(target_path);
  }

  // Remove "/./" and "/../" to make this more friendly to path-allowlist-based
  // sandboxes.
  return MakeAbsoluteFilePathNoResolveSymbolicLinks(target_path);
}

bool GetPosixFilePermissions(const FilePath& path, int* mode) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  DCHECK(mode);

  stat_wrapper_t file_info;
  // Uses stat(), because on symbolic link, lstat() does not return valid
  // permission bits in st_mode
  if (File::Stat(path.value().c_str(), &file_info) != 0)
    return false;

  *mode = file_info.st_mode & FILE_PERMISSION_MASK;
  return true;
}

bool SetPosixFilePermissions(const FilePath& path,
                             int mode) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  DCHECK_EQ(mode & ~FILE_PERMISSION_MASK, 0);

  // Calls stat() so that we can preserve the higher bits like S_ISGID.
  stat_wrapper_t stat_buf;
  if (File::Stat(path.value().c_str(), &stat_buf) != 0)
    return false;

  // Clears the existing permission bits, and adds the new ones.
  // The casting here is because the Android NDK does not declare `st_mode` as a
  // `mode_t`.
  mode_t updated_mode_bits = static_cast<mode_t>(stat_buf.st_mode);
  updated_mode_bits &= static_cast<mode_t>(~FILE_PERMISSION_MASK);
  updated_mode_bits |= mode & FILE_PERMISSION_MASK;

  if (HANDLE_EINTR(chmod(path.value().c_str(), updated_mode_bits)) != 0)
    return false;

  return true;
}

/*
TODO(chokobole):
bool ExecutableExistsInPath(const std::string& executable) {
  std::string_view path;
  if (!Environment::Get("PATH", &path)) {
    LOG(ERROR) << "No $PATH variable. Assuming no " << executable << ".";
    return false;
  }

  for (const std::string_view& cur_path :
       SplitStringView(path, ":", KEEP_WHITESPACE, SPLIT_WANT_NONEMPTY)) {
    FilePath file(cur_path);
    int permissions;
    if (GetPosixFilePermissions(file.Append(executable), &permissions) &&
        (permissions & FILE_PERMISSION_EXECUTE_BY_USER))
      return true;
  }
  return false;
}
*/

#endif  // !BUILDFLAG(IS_FUCHSIA)

#if !BUILDFLAG(IS_APPLE)
// This is implemented in file_util_mac.mm for Mac.
bool GetTempDir(FilePath* path) {
  const char* tmp = getenv("TMPDIR");
  if (tmp) {
    *path = FilePath(tmp);
    return true;
  }

#if BUILDFLAG(IS_ANDROID)
  return PathService::Get(DIR_CACHE, path);
#else
  *path = FilePath("/tmp");
  return true;
#endif
}
#endif  // !BUILDFLAG(IS_APPLE)

#if !BUILDFLAG(IS_APPLE)  // Mac implementation is in file_util_mac.mm.
FilePath GetHomeDir() {
#if BUILDFLAG(IS_CHROMEOS)
  if (SysInfo::IsRunningOnChromeOS()) {
    // On Chrome OS chrome::DIR_USER_DATA is overridden with a primary user
    // homedir once it becomes available. Return / as the safe option.
    return FilePath("/");
  }
#endif

  const char* home_dir = getenv("HOME");
  if (home_dir && home_dir[0])
    return FilePath(home_dir);

#if BUILDFLAG(IS_ANDROID)
  DLOG(WARNING) << "OS_ANDROID: Home directory lookup not yet implemented.";
#endif

  FilePath rv;
  if (GetTempDir(&rv))
    return rv;

  // Last resort.
  return FilePath("/tmp");
}
#endif  // !BUILDFLAG(IS_APPLE)

File CreateAndOpenTemporaryFileInDir(const FilePath& dir, FilePath* temp_file) {
  // For call to close() inside ScopedFD.
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  ScopedFD fd = CreateAndOpenFdForTemporaryFileInDir(dir, temp_file);
  return fd.is_valid() ? File(std::move(fd)) : File(File::GetLastFileError());
}

bool CreateTemporaryFileInDir(const FilePath& dir, FilePath* temp_file) {
  // For call to close() inside ScopedFD.
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  ScopedFD fd = CreateAndOpenFdForTemporaryFileInDir(dir, temp_file);
  return fd.is_valid();
}

FilePath FormatTemporaryFileName(std::string_view identifier) {
#if BUILDFLAG(IS_APPLE)
  std::string_view prefix = base::mac::BaseBundleID();
#else
  std::string_view prefix = "org.kroma.Tachyon";
#endif
  return FilePath(absl::StrCat(".", prefix, ".", identifier));
}

ScopedFILE CreateAndOpenTemporaryStreamInDir(const FilePath& dir,
                                             FilePath* path) {
  ScopedFD scoped_fd = CreateAndOpenFdForTemporaryFileInDir(dir, path);
  if (!scoped_fd.is_valid())
    return nullptr;

  int fd = scoped_fd.release();
  FILE* file = fdopen(fd, "a+");
  if (!file)
    close(fd);
  return ScopedFILE(file);
}

static bool CreateTemporaryDirInDirImpl(const FilePath& base_dir,
                                        const FilePath& name_tmpl,
                                        FilePath* new_dir) {
  /*
  TODO(chokobole):
  ScopedBlockingCall scoped_blocking_call(
      FROM_HERE, BlockingType::MAY_BLOCK);  // For call to mkdtemp().
  */
  DCHECK(EndsWith(name_tmpl.value(), "XXXXXX"))
      << "Directory name template must end with \"XXXXXX\".";

  FilePath sub_dir = base_dir.Append(name_tmpl);
  std::string sub_dir_string = sub_dir.value();

  // this should be OK since mkdtemp just replaces characters in place
  char* buffer = const_cast<char*>(sub_dir_string.c_str());
  char* dtemp = mkdtemp(buffer);
  if (!dtemp) {
    DPLOG(ERROR) << "mkdtemp";
    return false;
  }
  *new_dir = FilePath(dtemp);
  return true;
}

bool CreateTemporaryDirInDir(const FilePath& base_dir,
                             const std::string& prefix,
                             FilePath* new_dir) {
  std::string mkdtemp_template = prefix;
  mkdtemp_template.append("XXXXXX");
  return CreateTemporaryDirInDirImpl(base_dir, FilePath(mkdtemp_template),
                                     new_dir);
}

bool CreateNewTempDirectory(const std::string& prefix,
                            FilePath* new_temp_path) {
  FilePath tmpdir;
  if (!GetTempDir(&tmpdir))
    return false;

  return CreateTemporaryDirInDirImpl(tmpdir, GetTempTemplate(), new_temp_path);
}

bool CreateDirectoryAndGetError(const FilePath& full_path,
                                File::Error* error) {
  /*
  TODO(chokobole):
  ScopedBlockingCall scoped_blocking_call(
      FROM_HERE, BlockingType::MAY_BLOCK);  // For call to mkdir().
  */
  std::vector<FilePath> subpaths;

  // Collect a list of all parent directories.
  FilePath last_path = full_path;
  subpaths.push_back(full_path);
  for (FilePath path = full_path.DirName();
       path.value() != last_path.value(); path = path.DirName()) {
    subpaths.push_back(path);
    last_path = path;
  }

  // Iterate through the parents and create the missing ones.
  for (const FilePath& subpath : base::Reversed(subpaths)) {
    if (DirectoryExists(subpath))
      continue;
    if (mkdir(subpath.value().c_str(), 0700) == 0)
      continue;
    // Mkdir failed, but it might have failed with EEXIST, or some other error
    // due to the directory appearing out of thin air. This can occur if
    // two processes are trying to create the same file system tree at the same
    // time. Check to see if it exists and make sure it is a directory.
    int saved_errno = errno;
    if (!DirectoryExists(subpath)) {
      if (error)
        *error = File::OSErrorToFileError(saved_errno);
      return false;
    }
  }
  return true;
}

// ReadFileToStringNonBlockingNonBlocking will read a file to a string. This
// method should only be used on files which are known to be non-blocking such
// as procfs or sysfs nodes. Additionally, the file is opened as O_NONBLOCK so
// it WILL NOT block even if opened on a blocking file. It will return true if
// the file read until EOF and it will return false otherwise, errno will remain
// set on error conditions. |ret| will be populated with the contents of the
// file.
bool ReadFileToStringNonBlocking(const base::FilePath& file, std::string* ret) {
  DCHECK(ret);
  ret->clear();

  base::ScopedFD fd(HANDLE_EINTR(
      open(file.value().c_str(), O_CLOEXEC | O_NONBLOCK | O_RDONLY)));
  if (!fd.is_valid()) {
    return false;
  }

  ssize_t bytes_read = 0;
  do {
    char buf[4096];
    bytes_read = HANDLE_EINTR(read(fd.get(), buf, sizeof(buf)));
    if (bytes_read < 0)
      return false;
    if (bytes_read > 0)
      ret->append(buf, static_cast<size_t>(bytes_read));
  } while (bytes_read > 0);

  return true;
}

bool NormalizeFilePath(const FilePath& path, FilePath* normalized_path) {
  FilePath real_path_result = MakeAbsoluteFilePath(path);
  if (real_path_result.empty())
    return false;

  // To be consistant with windows, fail if |real_path_result| is a
  // directory.
  if (DirectoryExists(real_path_result))
    return false;

  *normalized_path = real_path_result;
  return true;
}

// TODO(rkc): Refactor GetFileInfo and FileEnumerator to handle symlinks
// correctly. http://code.google.com/p/chromium-os/issues/detail?id=15948
bool IsLink(const FilePath& file_path) {
  stat_wrapper_t st;
  // If we can't lstat the file, it's safe to assume that the file won't at
  // least be a 'followable' link.
  if (File::Lstat(file_path.value().c_str(), &st) != 0)
    return false;
  return S_ISLNK(st.st_mode);
}

bool GetFileInfo(const FilePath& file_path, File::Info* results) {
  stat_wrapper_t file_info;
#if BUILDFLAG(IS_ANDROID)
  if (file_path.IsContentUri()) {
    File file = OpenContentUriForRead(file_path);
    if (!file.IsValid())
      return false;
    return file.GetInfo(results);
  } else {
#endif  // BUILDFLAG(IS_ANDROID)
    if (File::Stat(file_path.value().c_str(), &file_info) != 0)
      return false;
#if BUILDFLAG(IS_ANDROID)
  }
#endif  // BUILDFLAG(IS_ANDROID)

  results->FromStat(file_info);
  return true;
}

FILE* OpenFile(const FilePath& filename, const char* mode) {
  // 'e' is unconditionally added below, so be sure there is not one already
  // present before a comma in |mode|.
  DCHECK(
      strchr(mode, 'e') == nullptr ||
      (strchr(mode, ',') != nullptr && strchr(mode, 'e') > strchr(mode, ',')));
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  FILE* result = nullptr;
#if BUILDFLAG(IS_APPLE)
  // macOS does not provide a mode character to set O_CLOEXEC; see
  // https://developer.apple.com/legacy/library/documentation/Darwin/Reference/ManPages/man3/fopen.3.html.
  const char* the_mode = mode;
#else
  std::string mode_with_e(AppendModeCharacter(mode, 'e'));
  const char* the_mode = mode_with_e.c_str();
#endif
  do {
    result = fopen(filename.value().c_str(), the_mode);
  } while (!result && errno == EINTR);
#if BUILDFLAG(IS_APPLE)
  // Mark the descriptor as close-on-exec.
  if (result)
    SetCloseOnExec(fileno(result));
#endif
  return result;
}

// NaCl doesn't implement system calls to open files directly.
#if !BUILDFLAG(IS_NACL)
FILE* FileToFILE(File file, const char* mode) {
  PlatformFile unowned = file.GetPlatformFile();
  FILE* stream = fdopen(file.TakePlatformFile(), mode);
  if (!stream)
    ScopedFD to_be_closed(unowned);
  return stream;
}

File FILEToFile(FILE* file_stream) {
  if (!file_stream)
    return File();

  PlatformFile fd = fileno(file_stream);
  DCHECK_NE(fd, -1);
  ScopedPlatformFile other_fd(HANDLE_EINTR(dup(fd)));
  if (!other_fd.is_valid())
    return File(File::GetLastFileError());
  return File(std::move(other_fd));
}
#endif  // !BUILDFLAG(IS_NACL)

int ReadFile(const FilePath& filename, char* data, int max_size) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  if (max_size < 0)
    return -1;
  int fd = HANDLE_EINTR(open(filename.value().c_str(), O_RDONLY));
  if (fd < 0)
    return -1;

  long bytes_read = HANDLE_EINTR(read(fd, data, static_cast<size_t>(max_size)));
  if (IGNORE_EINTR(close(fd)) < 0)
    return -1;
  return checked_cast<int>(bytes_read);
}

int WriteFile(const FilePath& filename, const char* data, int size) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  if (size < 0)
    return -1;
  int fd = HANDLE_EINTR(creat(filename.value().c_str(), 0666));
  if (fd < 0)
    return -1;

  int bytes_written =
      WriteFileDescriptor(fd, std::string_view(data, static_cast<size_t>(size)))
          ? size
          : -1;
  if (IGNORE_EINTR(close(fd)) < 0)
    return -1;
  return bytes_written;
}

bool WriteFileDescriptor(int fd, absl::Span<const uint8_t> data) {
  // Allow for partial writes.
  ssize_t bytes_written_total = 0;
  ssize_t size = checked_cast<ssize_t>(data.size());
  for (ssize_t bytes_written_partial = 0; bytes_written_total < size;
       bytes_written_total += bytes_written_partial) {
    bytes_written_partial =
        HANDLE_EINTR(write(fd, data.data() + bytes_written_total,
                           static_cast<size_t>(size - bytes_written_total)));
    if (bytes_written_partial < 0)
      return false;
  }

  return true;
}

bool WriteFileDescriptor(int fd, std::string_view data) {
  return WriteFileDescriptor(fd, absl::Span<const uint8_t>(
    reinterpret_cast<const uint8_t*>(data.data()), data.length()));
}

bool AllocateFileRegion(File* file, int64_t offset, size_t size) {
  DCHECK(file);

  // Explicitly extend |file| to the maximum size. Zeros will fill the new
  // space. It is assumed that the existing file is fully realized as
  // otherwise the entire file would have to be read and possibly written.
  const int64_t original_file_len = file->GetLength();
  if (original_file_len < 0) {
    DPLOG(ERROR) << "fstat " << file->GetPlatformFile();
    return false;
  }

  // Increase the actual length of the file, if necessary. This can fail if
  // the disk is full and the OS doesn't support sparse files.
  const int64_t new_file_len = offset + static_cast<int64_t>(size);
  // If the first condition fails, the cast on the previous line was invalid
  // (though not UB).
  if (!IsValueInRangeForNumericType<int64_t>(size) ||
      !IsValueInRangeForNumericType<off_t>(size) ||
      !IsValueInRangeForNumericType<off_t>(new_file_len) ||
      !file->SetLength(std::max(original_file_len, new_file_len))) {
    DPLOG(ERROR) << "ftruncate " << file->GetPlatformFile();
    return false;
  }

  // Realize the extent of the file so that it can't fail (and crash) later
  // when trying to write to a memory page that can't be created. This can
  // fail if the disk is full and the file is sparse.

  // First try the more effective platform-specific way of allocating the disk
  // space. It can fail because the filesystem doesn't support it. In that case,
  // use the manual method below.

#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS)
  if (HANDLE_EINTR(fallocate(file->GetPlatformFile(), 0, offset,
                             static_cast<off_t>(size))) != -1)
    return true;
  DPLOG(ERROR) << "fallocate";
#elif BUILDFLAG(IS_APPLE)
  // MacOS doesn't support fallocate even though their new APFS filesystem
  // does support sparse files. It does, however, have the functionality
  // available via fcntl.
  // See also: https://openradar.appspot.com/32720223
  fstore_t params = {F_ALLOCATEALL, F_PEOFPOSMODE, offset,
                     static_cast<off_t>(size), 0};
  if (fcntl(file->GetPlatformFile(), F_PREALLOCATE, &params) != -1)
    return true;
  DPLOG(ERROR) << "F_PREALLOCATE";
#endif

  // Manually realize the extended file by writing bytes to it at intervals.
  blksize_t block_size = 512;  // Start with something safe.
  stat_wrapper_t statbuf;
  if (File::Fstat(file->GetPlatformFile(), &statbuf) == 0 &&
      statbuf.st_blksize > 0 && base::bits::IsPowerOfTwo(statbuf.st_blksize)) {
    block_size = static_cast<blksize_t>(statbuf.st_blksize);
  }

  // Write starting at the next block boundary after the old file length.
  const int64_t extension_start = checked_cast<int64_t>(base::bits::AlignUp(
      static_cast<size_t>(original_file_len), static_cast<size_t>(block_size)));
  for (int64_t i = extension_start; i < new_file_len; i += block_size) {
    char existing_byte;
    if (HANDLE_EINTR(pread(file->GetPlatformFile(), &existing_byte, 1,
                           static_cast<off_t>(i))) != 1) {
      return false;  // Can't read? Not viable.
    }
    if (existing_byte != 0) {
      continue;  // Block has data so must already exist.
    }
    if (HANDLE_EINTR(pwrite(file->GetPlatformFile(), &existing_byte, 1,
                            static_cast<off_t>(i))) != 1) {
      return false;  // Can't write? Not viable.
    }
  }

  return true;
}

bool AppendToFile(const FilePath& filename, absl::Span<const uint8_t> data) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  bool ret = true;
  int fd = HANDLE_EINTR(open(filename.value().c_str(), O_WRONLY | O_APPEND));
  if (fd < 0) {
    VPLOG(1) << "Unable to create file " << filename.value();
    return false;
  }

  // This call will either write all of the data or return false.
  if (!WriteFileDescriptor(fd, data)) {
    VPLOG(1) << "Error while writing to file " << filename.value();
    ret = false;
  }

  if (IGNORE_EINTR(close(fd)) < 0) {
    VPLOG(1) << "Error while closing file " << filename.value();
    return false;
  }

  return ret;
}

bool AppendToFile(const FilePath& filename, std::string_view data) {
  return AppendToFile(filename, absl::Span<const uint8_t>(
    reinterpret_cast<const uint8_t*>(data.data()), data.length()));
}

bool GetCurrentDirectory(FilePath* dir) {
  // getcwd can return ENOENT, which implies it checks against the disk.
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);

  char system_buffer[PATH_MAX] = "";
  if (!getcwd(system_buffer, sizeof(system_buffer))) {
    return false;
  }
  *dir = FilePath(system_buffer);
  return true;
}

bool SetCurrentDirectory(const FilePath& path) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  return chdir(path.value().c_str()) == 0;
}

#if BUILDFLAG(IS_MAC)
bool VerifyPathControlledByUser(const FilePath& base,
                                const FilePath& path,
                                uid_t owner_uid,
                                const std::set<gid_t>& group_gids) {
  if (base != path && !base.IsParent(path)) {
     DLOG(ERROR) << "|base| must be a subdirectory of |path|.  base = \""
                 << base.value() << "\", path = \"" << path.value() << "\"";
     return false;
  }

  std::vector<std::string> base_components = base.GetComponents();
  std::vector<std::string> path_components = path.GetComponents();
  std::vector<std::string>::const_iterator ib, ip;
  for (ib = base_components.begin(), ip = path_components.begin();
       ib != base_components.end(); ++ib, ++ip) {
    // |base| must be a subpath of |path|, so all components should match.
    // If these CHECKs fail, look at the test that base is a parent of
    // path at the top of this function.
    DCHECK(ip != path_components.end());
    DCHECK(*ip == *ib);
  }

  FilePath current_path = base;
  if (!VerifySpecificPathControlledByUser(current_path, owner_uid, group_gids))
    return false;

  for (; ip != path_components.end(); ++ip) {
    current_path = current_path.Append(*ip);
    if (!VerifySpecificPathControlledByUser(
            current_path, owner_uid, group_gids))
      return false;
  }
  return true;
}

bool VerifyPathControlledByAdmin(const FilePath& path) {
  const unsigned kRootUid = 0;
  const FilePath kFileSystemRoot("/");

  // The name of the administrator group on mac os.
  const char* const kAdminGroupNames[] = {
    "admin",
    "wheel"
  };

  // Reading the groups database may touch the file system.
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);

  std::set<gid_t> allowed_group_ids;
  for (int i = 0, ie = std::size(kAdminGroupNames); i < ie; ++i) {
    struct group *group_record = getgrnam(kAdminGroupNames[i]);
    if (!group_record) {
      DPLOG(ERROR) << "Could not get the group ID of group \""
                   << kAdminGroupNames[i] << "\".";
      continue;
    }

    allowed_group_ids.insert(group_record->gr_gid);
  }

  return VerifyPathControlledByUser(
      kFileSystemRoot, path, kRootUid, allowed_group_ids);
}
#endif  // BUILDFLAG(IS_MAC)

int GetMaximumPathComponentLength(const FilePath& path) {
#if BUILDFLAG(IS_FUCHSIA)
  // Return a value we do not expect anyone ever to reach, but which is small
  // enough to guard against e.g. bugs causing multi-megabyte paths.
  return 1024;
#else
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  return saturated_cast<int>(pathconf(path.value().c_str(), _PC_NAME_MAX));
#endif
}

#if !BUILDFLAG(IS_ANDROID)
// This is implemented in file_util_android.cc for that platform.
bool GetShmemTempDir(bool executable, FilePath* path) {
#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_AIX)
  bool disable_dev_shm = false;
// #if !BUILDFLAG(IS_CHROMEOS)
//   disable_dev_shm = CommandLine::ForCurrentProcess()->HasSwitch(
//       switches::kDisableDevShmUsage);
// #endif
  bool use_dev_shm = true;
  if (executable) {
    static const bool s_dev_shm_executable =
        IsPathExecutable(FilePath("/dev/shm"));
    use_dev_shm = s_dev_shm_executable;
  }
  if (use_dev_shm && !disable_dev_shm) {
    *path = FilePath("/dev/shm");
    return true;
  }
#endif  // BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_AIX)
  return GetTempDir(path);
}
#endif  // !BUILDFLAG(IS_ANDROID)

#if !BUILDFLAG(IS_APPLE)
// Mac has its own implementation, this is for all other Posix systems.
bool CopyFile(const FilePath& from_path, const FilePath& to_path) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  File infile;
#if BUILDFLAG(IS_ANDROID)
  if (from_path.IsContentUri()) {
    infile = OpenContentUriForRead(from_path);
  } else {
    infile = File(from_path, File::FLAG_OPEN | File::FLAG_READ);
  }
#else
  infile = File(from_path, File::FLAG_OPEN | File::FLAG_READ);
#endif
  if (!infile.IsValid())
    return false;

  File outfile(to_path, File::FLAG_WRITE | File::FLAG_CREATE_ALWAYS);
  if (!outfile.IsValid())
    return false;

  return CopyFileContents(infile, outfile);
}
#endif  // !BUILDFLAG(IS_APPLE)

bool PreReadFile(const FilePath& file_path,
                 bool is_executable,
                 int64_t max_bytes) {
  DCHECK_GE(max_bytes, 0);

  // posix_fadvise() is only available in the Android NDK in API 21+. Older
  // versions may have the required kernel support, but don't have enough usage
  // to justify backporting.
#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) || \
    (BUILDFLAG(IS_ANDROID) && __ANDROID_API__ >= 21)
  File file(file_path, File::FLAG_OPEN | File::FLAG_READ);
  if (!file.IsValid())
    return false;

  if (max_bytes == 0) {
    // fadvise() pre-fetches the entire file when given a zero length.
    return true;
  }

  const PlatformFile fd = file.GetPlatformFile();
  const ::off_t len = base::saturated_cast<::off_t>(max_bytes);
  return posix_fadvise(fd, /*offset=*/0, len, POSIX_FADV_WILLNEED) == 0;
#elif BUILDFLAG(IS_APPLE)
  File file(file_path, File::FLAG_OPEN | File::FLAG_READ);
  if (!file.IsValid())
    return false;

  if (max_bytes == 0) {
    // fcntl(F_RDADVISE) fails when given a zero length.
    return true;
  }

  const PlatformFile fd = file.GetPlatformFile();
  ::radvisory read_advise_data = {
      .ra_offset = 0, .ra_count = base::saturated_cast<int>(max_bytes)};
  return fcntl(fd, F_RDADVISE, &read_advise_data) != -1;
#else
  return internal::PreReadFileSlow(file_path, max_bytes);
#endif  // BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) ||
        // (BUILDFLAG(IS_ANDROID) &&
        // __ANDROID_API__ >= 21)
}

// -----------------------------------------------------------------------------

namespace internal {

bool MoveUnsafe(const FilePath& from_path, const FilePath& to_path) {
  // TODO(chokobole):
  // ScopedBlockingCall scoped_blocking_call(FROM_HERE, BlockingType::MAY_BLOCK);
  // Windows compatibility: if |to_path| exists, |from_path| and |to_path|
  // must be the same type, either both files, or both directories.
  stat_wrapper_t to_file_info;
  if (File::Stat(to_path.value().c_str(), &to_file_info) == 0) {
    stat_wrapper_t from_file_info;
    if (File::Stat(from_path.value().c_str(), &from_file_info) != 0)
      return false;
    if (S_ISDIR(to_file_info.st_mode) != S_ISDIR(from_file_info.st_mode))
      return false;
  }

  if (rename(from_path.value().c_str(), to_path.value().c_str()) == 0)
    return true;

  if (!CopyDirectory(from_path, to_path, true))
    return false;

  DeletePathRecursively(from_path);
  return true;
}

#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_ANDROID)
bool CopyFileContentsWithSendfile(File& infile,
                                  File& outfile,
                                  bool& retry_slow) {
  DCHECK(infile.IsValid());
  stat_wrapper_t in_file_info;
  retry_slow = false;

  if (base::File::Fstat(infile.GetPlatformFile(), &in_file_info)) {
    return false;
  }

  int64_t file_size = in_file_info.st_size;
  if (file_size < 0)
    return false;
  if (file_size == 0) {
    // Non-regular files can return a file size of 0, things such as pipes,
    // sockets, etc. Additionally, kernel seq_files(most procfs files) will also
    // return 0 while still reporting as a regular file. Unfortunately, in some
    // of these situations there are easy ways to detect them, in others there
    // are not. No extra syscalls are needed if it's not a regular file.
    //
    // Because any attempt to detect it would likely require another syscall,
    // let's just fall back to a slow copy which will invoke a single read(2) to
    // determine if the file has contents or if it's really a zero length file.
    retry_slow = true;
    return false;
  }

  size_t copied = 0;
  ssize_t res = 0;
  do {
    // Don't specify an offset and the kernel will begin reading/writing to the
    // current file offsets.
    res = HANDLE_EINTR(sendfile(
        outfile.GetPlatformFile(), infile.GetPlatformFile(), /*offset=*/nullptr,
        /*length=*/static_cast<size_t>(file_size) - copied));
    if (res <= 0) {
      break;
    }

    copied += static_cast<size_t>(res);
  } while (copied < static_cast<size_t>(file_size));

  // Fallback on non-fatal error cases. None of these errors can happen after
  // data has started copying, a check is included for good measure. As a result
  // file sizes and file offsets will not have changed. A slow fallback and
  // proceed without issues.
  retry_slow = (copied == 0 && res < 0 &&
                (errno == EINVAL || errno == ENOSYS || errno == EPERM));

  return res >= 0;
}
#endif  // BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) ||
        // BUILDFLAG(IS_ANDROID)

}  // namespace internal

#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_AIX)
TACHYON_EXPORT bool IsPathExecutable(const FilePath& path) {
  bool result = false;
  FilePath tmp_file_path;

  ScopedFD fd = CreateAndOpenFdForTemporaryFileInDir(path, &tmp_file_path);
  if (fd.is_valid()) {
    DeleteFile(tmp_file_path);
    long sysconf_result = sysconf(_SC_PAGESIZE);
    CHECK_GE(sysconf_result, 0);
    size_t pagesize = static_cast<size_t>(sysconf_result);
    CHECK_GE(sizeof(pagesize), sizeof(sysconf_result));
    void* mapping = mmap(nullptr, pagesize, PROT_READ, MAP_SHARED, fd.get(), 0);
    if (mapping != MAP_FAILED) {
      if (HANDLE_EINTR(mprotect(mapping, pagesize, PROT_READ | PROT_EXEC)) == 0)
        result = true;
      munmap(mapping, pagesize);
    }
  }
  return result;
}
#endif  // BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_AIX)

}  // namespace tachyon::base
