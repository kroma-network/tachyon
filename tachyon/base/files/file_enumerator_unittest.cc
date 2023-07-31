// Copyright 2017 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/files/file_enumerator.h"

#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "tachyon/base/containers/circular_deque.h"
#include "tachyon/base/files/file_path.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/files/scoped_temp_dir.h"
#include "tachyon/base/logging.h"
#include "tachyon/build/build_config.h"

using testing::ElementsAre;
using testing::IsEmpty;
using testing::UnorderedElementsAre;

namespace tachyon::base {
namespace {

const std::string kEmptyPattern;

const std::vector<FileEnumerator::FolderSearchPolicy> kFolderSearchPolicies{
    FileEnumerator::FolderSearchPolicy::MATCH_ONLY,
    FileEnumerator::FolderSearchPolicy::ALL};

struct TestFile {
  TestFile(const char* file_name, const char* c)
      : path(file_name), contents(c) {}

  TestFile(const char* directory,
           const char* file_name,
           const char* c)
      : path(FilePath(directory).Append(file_name)), contents(c) {}

  const FilePath path;
  const std::string contents;
  File::Info info;
  bool found = false;
};

struct TestDirectory {
  explicit TestDirectory(const char* n) : name(n) {}
  const FilePath name;
  File::Info info;
  bool found = false;
};

void CheckModificationTime(const FileEnumerator::FileInfo& actual,
                           Time expected_last_modified_time) {
  /*
  TODO(chokobole):
#if BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
  // On POSIX, GetLastModifiedTime() rounds down to the second, but
  // File::GetInfo() does not.
  Time::Exploded exploded;
  expected_last_modified_time.UTCExplode(&exploded);
  exploded.millisecond = 0;
  EXPECT_TRUE(Time::FromUTCExploded(exploded, &expected_last_modified_time));
#endif
  EXPECT_EQ(actual.GetLastModifiedTime(), expected_last_modified_time);
  */
}

void CheckFileAgainstInfo(const FileEnumerator::FileInfo& actual,
                          TestFile& expected) {
  EXPECT_FALSE(expected.found)
      << "Got " << expected.path.BaseName().value() << " twice";
  expected.found = true;
  EXPECT_EQ(actual.GetSize(), int64_t(expected.contents.size()));
  CheckModificationTime(actual, expected.info.last_modified);
}

void CheckDirectoryAgainstInfo(const FileEnumerator::FileInfo& actual,
                               TestDirectory& expected) {
  EXPECT_FALSE(expected.found) << "Got " << expected.name.value() << " twice";
  expected.found = true;
  CheckModificationTime(actual, expected.info.last_modified);
}

circular_deque<FilePath> RunEnumerator(
    const FilePath& root_path,
    bool recursive,
    int file_type,
    const std::string& pattern,
    FileEnumerator::FolderSearchPolicy folder_search_policy) {
  circular_deque<FilePath> rv;
  FileEnumerator enumerator(root_path, recursive, file_type, pattern,
                            folder_search_policy,
                            FileEnumerator::ErrorPolicy::IGNORE_ERRORS);
  for (auto file = enumerator.Next(); !file.empty(); file = enumerator.Next())
    rv.emplace_back(std::move(file));
  return rv;
}

bool CreateDummyFile(const FilePath& path) {
  return WriteFile(path, "42", sizeof("42")) == sizeof("42");
}

bool GetFileInfo(const FilePath& file_path, File::Info& info) {
  // FLAG_WIN_BACKUP_SEMANTICS: Needed to open directories on Windows.
  File f(file_path,
         File::FLAG_OPEN | File::FLAG_READ | File::FLAG_WIN_BACKUP_SEMANTICS);
  if (!f.IsValid()) {
    LOG(ERROR) << "Could not open " << file_path.value() << ": "
               << File::ErrorToString(f.error_details());
    return false;
  }
  if (!f.GetInfo(&info)) {
    std::string last_error = File::ErrorToString(File::GetLastFileError());
    LOG(ERROR) << "Could not get info about " << file_path.value() << ": "
               << last_error;
    return false;
  }

  return true;
}

void SetUpTestFiles(const ScopedTempDir& temp_dir,
                    std::vector<TestFile>& files) {
  for (TestFile& file : files) {
    const FilePath file_path = temp_dir.GetPath().Append(file.path);
    ASSERT_TRUE(WriteFile(file_path, file.contents));
    ASSERT_TRUE(GetFileInfo(file_path, file.info));
  }
}

}  // namespace

TEST(FileEnumerator, NotExistingPath) {
  const FilePath path = FilePath("some_not_existing_path");
  ASSERT_FALSE(PathExists(path));

  for (auto policy : kFolderSearchPolicies) {
    const auto files = RunEnumerator(
        path, true, FileEnumerator::FILES | FileEnumerator::DIRECTORIES,
        "", policy);
    EXPECT_THAT(files, IsEmpty());
  }
}

TEST(FileEnumerator, EmptyFolder) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  for (auto policy : kFolderSearchPolicies) {
    const auto files =
        RunEnumerator(temp_dir.GetPath(), true,
                      FileEnumerator::FILES | FileEnumerator::DIRECTORIES,
                      kEmptyPattern, policy);
    EXPECT_THAT(files, IsEmpty());
  }
}

TEST(FileEnumerator, SingleFileInFolderForFileSearch) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath& path = temp_dir.GetPath();
  const FilePath file = path.Append("test.txt");
  ASSERT_TRUE(CreateDummyFile(file));

  for (auto policy : kFolderSearchPolicies) {
    const auto files = RunEnumerator(
        temp_dir.GetPath(), true, FileEnumerator::FILES, kEmptyPattern, policy);
    EXPECT_THAT(files, ElementsAre(file));
  }
}

TEST(FileEnumerator, SingleFileInFolderForDirSearch) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath& path = temp_dir.GetPath();
  ASSERT_TRUE(CreateDummyFile(path.Append("test.txt")));

  for (auto policy : kFolderSearchPolicies) {
    const auto files = RunEnumerator(path, true, FileEnumerator::DIRECTORIES,
                                     kEmptyPattern, policy);
    EXPECT_THAT(files, IsEmpty());
  }
}

TEST(FileEnumerator, SingleFileInFolderWithFiltering) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath& path = temp_dir.GetPath();
  const FilePath file = path.Append("test.txt");
  ASSERT_TRUE(CreateDummyFile(file));

  for (auto policy : kFolderSearchPolicies) {
    auto files = RunEnumerator(path, true, FileEnumerator::FILES,
                               "*.txt", policy);
    EXPECT_THAT(files, ElementsAre(file));

    files = RunEnumerator(path, true, FileEnumerator::FILES,
                          "*.pdf", policy);
    EXPECT_THAT(files, IsEmpty());
  }
}

TEST(FileEnumerator, TwoFilesInFolder) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath& path = temp_dir.GetPath();
  const FilePath foo_txt = path.Append("foo.txt");
  const FilePath bar_txt = path.Append("bar.txt");
  ASSERT_TRUE(CreateDummyFile(foo_txt));
  ASSERT_TRUE(CreateDummyFile(bar_txt));

  for (auto policy : kFolderSearchPolicies) {
    auto files = RunEnumerator(path, true, FileEnumerator::FILES,
                               "*.txt", policy);
    EXPECT_THAT(files, UnorderedElementsAre(foo_txt, bar_txt));

    files = RunEnumerator(path, true, FileEnumerator::FILES,
                          "foo*", policy);
    EXPECT_THAT(files, ElementsAre(foo_txt));

    files = RunEnumerator(path, true, FileEnumerator::FILES,
                          "*.pdf", policy);
    EXPECT_THAT(files, IsEmpty());

    files =
        RunEnumerator(path, true, FileEnumerator::FILES, kEmptyPattern, policy);
    EXPECT_THAT(files, UnorderedElementsAre(foo_txt, bar_txt));
  }
}

TEST(FileEnumerator, SingleFolderInFolderForFileSearch) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath& path = temp_dir.GetPath();

  ScopedTempDir temp_subdir;
  ASSERT_TRUE(temp_subdir.CreateUniqueTempDirUnderPath(path));

  for (auto policy : kFolderSearchPolicies) {
    const auto files =
        RunEnumerator(path, true, FileEnumerator::FILES, kEmptyPattern, policy);
    EXPECT_THAT(files, IsEmpty());
  }
}

TEST(FileEnumerator, SingleFolderInFolderForDirSearch) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath& path = temp_dir.GetPath();

  ScopedTempDir temp_subdir;
  ASSERT_TRUE(temp_subdir.CreateUniqueTempDirUnderPath(path));

  for (auto policy : kFolderSearchPolicies) {
    const auto files = RunEnumerator(path, true, FileEnumerator::DIRECTORIES,
                                     kEmptyPattern, policy);
    EXPECT_THAT(files, ElementsAre(temp_subdir.GetPath()));
  }
}

TEST(FileEnumerator, TwoFoldersInFolder) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath& path = temp_dir.GetPath();

  const FilePath subdir_foo = path.Append("foo");
  const FilePath subdir_bar = path.Append("bar");
  ASSERT_TRUE(CreateDirectory(subdir_foo));
  ASSERT_TRUE(CreateDirectory(subdir_bar));

  for (auto policy : kFolderSearchPolicies) {
    auto files = RunEnumerator(path, true, FileEnumerator::DIRECTORIES,
                               kEmptyPattern, policy);
    EXPECT_THAT(files, UnorderedElementsAre(subdir_foo, subdir_bar));

    files = RunEnumerator(path, true, FileEnumerator::DIRECTORIES,
                          "foo", policy);
    EXPECT_THAT(files, ElementsAre(subdir_foo));
  }
}

TEST(FileEnumerator, FolderAndFileInFolder) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath& path = temp_dir.GetPath();

  ScopedTempDir temp_subdir;
  ASSERT_TRUE(temp_subdir.CreateUniqueTempDirUnderPath(path));
  const FilePath file = path.Append("test.txt");
  ASSERT_TRUE(CreateDummyFile(file));

  for (auto policy : kFolderSearchPolicies) {
    auto files =
        RunEnumerator(path, true, FileEnumerator::FILES, kEmptyPattern, policy);
    EXPECT_THAT(files, ElementsAre(file));

    files = RunEnumerator(path, true, FileEnumerator::DIRECTORIES,
                          kEmptyPattern, policy);
    EXPECT_THAT(files, ElementsAre(temp_subdir.GetPath()));

    files = RunEnumerator(path, true,
                          FileEnumerator::FILES | FileEnumerator::DIRECTORIES,
                          kEmptyPattern, policy);
    EXPECT_THAT(files, UnorderedElementsAre(file, temp_subdir.GetPath()));
  }
}

TEST(FileEnumerator, FilesInParentFolderAlwaysFirst) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath& path = temp_dir.GetPath();

  ScopedTempDir temp_subdir;
  ASSERT_TRUE(temp_subdir.CreateUniqueTempDirUnderPath(path));
  const FilePath foo_txt = path.Append("foo.txt");
  const FilePath bar_txt = temp_subdir.GetPath().Append("bar.txt");
  ASSERT_TRUE(CreateDummyFile(foo_txt));
  ASSERT_TRUE(CreateDummyFile(bar_txt));

  for (auto policy : kFolderSearchPolicies) {
    const auto files =
        RunEnumerator(path, true, FileEnumerator::FILES, kEmptyPattern, policy);
    EXPECT_THAT(files, ElementsAre(foo_txt, bar_txt));
  }
}

TEST(FileEnumerator, FileInSubfolder) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath subdir = temp_dir.GetPath().Append("subdir");
  ASSERT_TRUE(CreateDirectory(subdir));

  const FilePath file = subdir.Append("test.txt");
  ASSERT_TRUE(CreateDummyFile(file));

  for (auto policy : kFolderSearchPolicies) {
    auto files = RunEnumerator(temp_dir.GetPath(), true, FileEnumerator::FILES,
                               kEmptyPattern, policy);
    EXPECT_THAT(files, ElementsAre(file));

    files = RunEnumerator(temp_dir.GetPath(), false, FileEnumerator::FILES,
                          kEmptyPattern, policy);
    EXPECT_THAT(files, IsEmpty());
  }
}

TEST(FileEnumerator, FilesInSubfoldersWithFiltering) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath test_txt = temp_dir.GetPath().Append("test.txt");
  const FilePath subdir_foo = temp_dir.GetPath().Append("foo_subdir");
  const FilePath subdir_bar = temp_dir.GetPath().Append("bar_subdir");
  const FilePath foo_test = subdir_foo.Append("test.txt");
  const FilePath foo_foo = subdir_foo.Append("foo.txt");
  const FilePath foo_bar = subdir_foo.Append("bar.txt");
  const FilePath bar_test = subdir_bar.Append("test.txt");
  const FilePath bar_foo = subdir_bar.Append("foo.txt");
  const FilePath bar_bar = subdir_bar.Append("bar.txt");
  ASSERT_TRUE(CreateDummyFile(test_txt));
  ASSERT_TRUE(CreateDirectory(subdir_foo));
  ASSERT_TRUE(CreateDirectory(subdir_bar));
  ASSERT_TRUE(CreateDummyFile(foo_test));
  ASSERT_TRUE(CreateDummyFile(foo_foo));
  ASSERT_TRUE(CreateDummyFile(foo_bar));
  ASSERT_TRUE(CreateDummyFile(bar_test));
  ASSERT_TRUE(CreateDummyFile(bar_foo));
  ASSERT_TRUE(CreateDummyFile(bar_bar));

  auto files =
      RunEnumerator(temp_dir.GetPath(), true,
                    FileEnumerator::FILES | FileEnumerator::DIRECTORIES,
                    "foo*",
                    FileEnumerator::FolderSearchPolicy::MATCH_ONLY);
  EXPECT_THAT(files,
              UnorderedElementsAre(subdir_foo, foo_test, foo_foo, foo_bar));

  files = RunEnumerator(temp_dir.GetPath(), true,
                        FileEnumerator::FILES | FileEnumerator::DIRECTORIES,
                        "foo*",
                        FileEnumerator::FolderSearchPolicy::ALL);
  EXPECT_THAT(files, UnorderedElementsAre(subdir_foo, foo_foo, bar_foo));
}

TEST(FileEnumerator, InvalidDirectory) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath test_file = temp_dir.GetPath().Append("test_file");
  ASSERT_TRUE(CreateDummyFile(test_file));

  // Attempt to enumerate entries at a regular file path.
  FileEnumerator enumerator(test_file, /*recursive=*/true,
                            FileEnumerator::FILES, kEmptyPattern,
                            FileEnumerator::FolderSearchPolicy::ALL,
                            FileEnumerator::ErrorPolicy::STOP_ENUMERATION);
  FilePath path = enumerator.Next();
  EXPECT_TRUE(path.empty());

  // Slightly different outcomes between Windows and POSIX.
#if BUILDFLAG(IS_WIN)
  EXPECT_EQ(File::Error::FILE_ERROR_FAILED, enumerator.GetError());
#else
  EXPECT_EQ(File::Error::FILE_ERROR_NOT_A_DIRECTORY, enumerator.GetError());
#endif
}

#if BUILDFLAG(IS_POSIX)
TEST(FileEnumerator, SymLinkLoops) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath subdir = temp_dir.GetPath().Append("subdir");
  ASSERT_TRUE(CreateDirectory(subdir));

  const FilePath file = subdir.Append("test.txt");
  ASSERT_TRUE(CreateDummyFile(file));

  const FilePath link = subdir.Append("link");
  ASSERT_TRUE(CreateSymbolicLink(temp_dir.GetPath(), link));

  auto files = RunEnumerator(
      temp_dir.GetPath(), true,
      FileEnumerator::FILES | FileEnumerator::DIRECTORIES, kEmptyPattern,
      FileEnumerator::FolderSearchPolicy::MATCH_ONLY);

  EXPECT_THAT(files, UnorderedElementsAre(subdir, link, file));

  files = RunEnumerator(subdir, true,
                        FileEnumerator::FILES | FileEnumerator::DIRECTORIES |
                            FileEnumerator::SHOW_SYM_LINKS,
                        kEmptyPattern,
                        FileEnumerator::FolderSearchPolicy::MATCH_ONLY);

  EXPECT_THAT(files, UnorderedElementsAre(link, file));
}
#endif

// Test FileEnumerator::GetInfo() on some files and ensure all the returned
// information is correct.
TEST(FileEnumerator, GetInfo) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  std::vector<TestFile> files = {
      TestFile("file1", "First"),
      TestFile("file2", "Second"),
      TestFile("file3", "Third-third-third")};
  SetUpTestFiles(temp_dir, files);

  FileEnumerator file_enumerator(temp_dir.GetPath(), false,
                                 FileEnumerator::FILES);
  while (!file_enumerator.Next().empty()) {
    auto info = file_enumerator.GetInfo();
    bool found = false;
    for (TestFile& file : files) {
      if (info.GetName() == file.path.BaseName()) {
        CheckFileAgainstInfo(info, file);
        found = true;
        break;
      }
    }

    EXPECT_TRUE(found) << "Got unexpected result " << info.GetName().value();
  }

  for (const TestFile& file : files) {
    EXPECT_TRUE(file.found)
        << "File " << file.path.value() << " was not returned";
  }
}

// Test that FileEnumerator::GetInfo() works when searching recursively. It also
// tests that it returns the correct information about directories.
TEST(FileEnumerator, GetInfoRecursive) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  TestDirectory directories[] = {TestDirectory("dir1"),
                                 TestDirectory("dir2"),
                                 TestDirectory("dir3"),
                                 TestDirectory("dirempty")};

  for (const TestDirectory& dir : directories) {
    const FilePath dir_path = temp_dir.GetPath().Append(dir.name);
    ASSERT_TRUE(CreateDirectory(dir_path));
  }

  std::vector<TestFile> files = {
      TestFile("dir1", "file1", "First"),
      TestFile("dir1", "file2", "Second"),
      TestFile("dir2", "fileA", "Third-third-3"),
      TestFile("dir3", ".file", "Dot")};
  SetUpTestFiles(temp_dir, files);

  // Get last-modification times for directories. Must be done after we create
  // all the files.
  for (TestDirectory& dir : directories) {
    const FilePath dir_path = temp_dir.GetPath().Append(dir.name);
    ASSERT_TRUE(GetFileInfo(dir_path, dir.info));
  }

  FileEnumerator file_enumerator(
      temp_dir.GetPath(), true,
      FileEnumerator::FILES | FileEnumerator::DIRECTORIES);
  while (!file_enumerator.Next().empty()) {
    auto info = file_enumerator.GetInfo();
    bool found = false;
    if (info.IsDirectory()) {
      for (TestDirectory& dir : directories) {
        if (info.GetName() == dir.name) {
          CheckDirectoryAgainstInfo(info, dir);
          found = true;
          break;
        }
      }
    } else {
      for (TestFile& file : files) {
        if (info.GetName() == file.path.BaseName()) {
          CheckFileAgainstInfo(info, file);
          found = true;
          break;
        }
      }
    }

    EXPECT_TRUE(found) << "Got unexpected result " << info.GetName().value();
  }

  for (const TestDirectory& dir : directories) {
    EXPECT_TRUE(dir.found) << "Directory " << dir.name.value()
                           << " was not returned";
  }
  for (const TestFile& file : files) {
    EXPECT_TRUE(file.found)
        << "File " << file.path.value() << " was not returned";
  }
}

#if BUILDFLAG(IS_FUCHSIA)
// FileEnumerator::GetInfo does not work correctly with INCLUDE_DOT_DOT.
// https://crbug.com/1106172
#elif BUILDFLAG(IS_WIN)
// Windows has a bug in their handling of ".."; they always report the file
// modification time of the current directory, not the parent directory. This is
// a bug in Windows, not us -- you can see it with the "dir" command (notice
// that the time of . and .. always match). Skip this test.
// https://crbug.com/1119546
#else
// Tests that FileEnumerator::GetInfo() returns the correct info for the ..
// directory.
TEST(FileEnumerator, GetInfoDotDot) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const char kSubdir[] = "subdir";
  const FilePath subdir_path = temp_dir.GetPath().Append(kSubdir);
  ASSERT_TRUE(CreateDirectory(subdir_path));

  std::vector<TestFile> files = {
      TestFile(kSubdir, "file1", "First"),
      TestFile(kSubdir, "file2", "Second"),
      TestFile(kSubdir, "file3", "Third-third-third")};
  SetUpTestFiles(temp_dir, files);

  TestDirectory dotdot("..");
  // test_dir/subdir/.. is just test_dir.
  ASSERT_TRUE(GetFileInfo(temp_dir.GetPath(), dotdot.info));

  FileEnumerator file_enumerator(subdir_path, false,
                                 FileEnumerator::FILES |
                                     FileEnumerator::DIRECTORIES |
                                     FileEnumerator::INCLUDE_DOT_DOT);
  while (!file_enumerator.Next().empty()) {
    auto info = file_enumerator.GetInfo();
    bool found = false;
    if (info.IsDirectory()) {
      EXPECT_EQ(info.GetName(), FilePath(".."));
      CheckDirectoryAgainstInfo(info, dotdot);
      found = true;
    } else {
      for (TestFile& file : files) {
        if (info.GetName() == file.path.BaseName()) {
          CheckFileAgainstInfo(info, file);
          found = true;
          break;
        }
      }
    }

    EXPECT_TRUE(found) << "Got unexpected result " << info.GetName().value();
  }

  EXPECT_TRUE(dotdot.found) << "Directory .. was not returned";

  for (const TestFile& file : files) {
    EXPECT_TRUE(file.found)
        << "File " << file.path.value() << " was not returned";
  }
}
#endif  // !BUILDFLAG(IS_FUCHSIA) && !BUILDFLAG(IS_WIN)

TEST(FileEnumerator, OnlyName) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());

  const FilePath& path = temp_dir.GetPath();

  // Add a directory and a file.
  ScopedTempDir temp_subdir;
  ASSERT_TRUE(temp_subdir.CreateUniqueTempDirUnderPath(path));
  const FilePath& subdir = temp_subdir.GetPath();
  const FilePath dummy_file = path.Append("a_file.txt");
  ASSERT_TRUE(CreateDummyFile(dummy_file));

  auto found_paths = RunEnumerator(
      path, /*recursive=*/false, FileEnumerator::FileType::NAMES_ONLY,
      std::string(), FileEnumerator::FolderSearchPolicy::MATCH_ONLY);
  EXPECT_THAT(found_paths, UnorderedElementsAre(subdir, dummy_file));
}

TEST(FileEnumerator, ForEach) {
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  const FilePath mock_path(temp_dir.GetPath());

  // Create a top-level directory, and 3 sub-directories, with 2 files within
  // each directory.
  for (const FilePath& path :
       {mock_path, mock_path.Append("1.2.3.4"),
        mock_path.Append("Download"),
        mock_path.Append("Install")}) {
    ASSERT_TRUE(CreateDirectory(path));
    for (std::string_view file_name : {"mock.executable", "mock.text"}) {
      ASSERT_TRUE(
          File(path.Append(file_name), File::FLAG_CREATE | File::FLAG_WRITE)
              .IsValid());
    }
  }

  const struct {
    bool recursive;
    int file_type;
    const int expected_invocation_count;
  } test_cases[] = {
      {false, FileEnumerator::FILES, 2},
      {true, FileEnumerator::FILES, 8},
      {false, FileEnumerator::DIRECTORIES, 3},
      {true, FileEnumerator::DIRECTORIES, 3},
      {false, FileEnumerator::FILES | FileEnumerator::DIRECTORIES, 5},
      {true, FileEnumerator::FILES | FileEnumerator::DIRECTORIES, 11},
  };

  for (const auto& test_case : test_cases) {
    int invocation_count = 0;

    FileEnumerator(mock_path, test_case.recursive, test_case.file_type)
        .ForEach([&invocation_count, &test_case](const FilePath& item) {
          ++invocation_count;
          if (invocation_count > test_case.expected_invocation_count) {
            ADD_FAILURE() << "Unexpected file/directory found: " << item << ": "
                          << invocation_count << ": "
                          << test_case.expected_invocation_count;
          }
        });

    EXPECT_EQ(invocation_count, test_case.expected_invocation_count);
  }
}

}  // namespace tachyon::base
