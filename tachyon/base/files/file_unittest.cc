// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/files/file.h"

#include <stdint.h>

#include <utility>

#include "tachyon/base/files/file_util.h"
// #include "tachyon/base/files/memory_mapped_file.h"
#include "tachyon/base/files/scoped_temp_dir.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/base/time/time.h"
#include "tachyon/build/build_config.h"
#include "tachyon/build/buildflag.h"
#include "gtest/gtest.h"

// TODO(chokobole):
// #if BUILDFLAG(ENABLE_BASE_TRACING)
// #include "third_party/perfetto/include/perfetto/test/traced_value_test_support.h"  // no-presubmit-check nogncheck
// #endif  // BUILDFLAG(ENABLE_BASE_TRACING)

#if BUILDFLAG(IS_WIN)
#include <windows.h>

#include "tachyon/base/environment.h"
#include "tachyon/base/strings/utf_string_conversions.h"
#include "tachyon/base/test/gtest_util.h"
#endif

namespace tachyon {

using base::File;
using base::FilePath;

TEST(FileTest, Create) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("create_file_1");

  {
    // Don't create a File at all.
    File file;
    EXPECT_FALSE(file.IsValid());
    EXPECT_EQ(base::File::FILE_ERROR_FAILED, file.error_details());

    File file2(base::File::FILE_ERROR_TOO_MANY_OPENED);
    EXPECT_FALSE(file2.IsValid());
    EXPECT_EQ(base::File::FILE_ERROR_TOO_MANY_OPENED, file2.error_details());
  }

  {
    // Open a file that doesn't exist.
    File file(file_path, base::File::FLAG_OPEN | base::File::FLAG_READ);
    EXPECT_FALSE(file.IsValid());
    EXPECT_EQ(base::File::FILE_ERROR_NOT_FOUND, file.error_details());
    EXPECT_EQ(base::File::FILE_ERROR_NOT_FOUND, base::File::GetLastFileError());
  }

  {
    // Open or create a file.
    File file(file_path, base::File::FLAG_OPEN_ALWAYS | base::File::FLAG_READ);
    EXPECT_TRUE(file.IsValid());
    EXPECT_TRUE(file.created());
    EXPECT_EQ(base::File::FILE_OK, file.error_details());
  }

  {
    // Open an existing file.
    File file(file_path, base::File::FLAG_OPEN | base::File::FLAG_READ);
    EXPECT_TRUE(file.IsValid());
    EXPECT_FALSE(file.created());
    EXPECT_EQ(base::File::FILE_OK, file.error_details());

    // This time verify closing the file.
    file.Close();
    EXPECT_FALSE(file.IsValid());
  }

  {
    // Open an existing file through Initialize
    File file;
    file.Initialize(file_path, base::File::FLAG_OPEN | base::File::FLAG_READ);
    EXPECT_TRUE(file.IsValid());
    EXPECT_FALSE(file.created());
    EXPECT_EQ(base::File::FILE_OK, file.error_details());

    // This time verify closing the file.
    file.Close();
    EXPECT_FALSE(file.IsValid());
  }

  {
    // Create a file that exists.
    File file(file_path, base::File::FLAG_CREATE | base::File::FLAG_READ);
    EXPECT_FALSE(file.IsValid());
    EXPECT_FALSE(file.created());
    EXPECT_EQ(base::File::FILE_ERROR_EXISTS, file.error_details());
    EXPECT_EQ(base::File::FILE_ERROR_EXISTS, base::File::GetLastFileError());
  }

  {
    // Create or overwrite a file.
    File file(file_path,
              base::File::FLAG_CREATE_ALWAYS | base::File::FLAG_WRITE);
    EXPECT_TRUE(file.IsValid());
    EXPECT_TRUE(file.created());
    EXPECT_EQ(base::File::FILE_OK, file.error_details());
  }

  {
    // Create a delete-on-close file.
    file_path = temp_dir.GetPath().Append("create_file_2");
    File file(file_path,
              base::File::FLAG_OPEN_ALWAYS | base::File::FLAG_READ |
                  base::File::FLAG_DELETE_ON_CLOSE);
    EXPECT_TRUE(file.IsValid());
    EXPECT_TRUE(file.created());
    EXPECT_EQ(base::File::FILE_OK, file.error_details());
  }

  EXPECT_FALSE(base::PathExists(file_path));
}

TEST(FileTest, SelfSwap) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("create_file_1");
  File file(file_path,
            base::File::FLAG_OPEN_ALWAYS | base::File::FLAG_DELETE_ON_CLOSE);
  std::swap(file, file);
  EXPECT_TRUE(file.IsValid());
}

TEST(FileTest, Async) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("create_file");

  {
    File file(file_path, base::File::FLAG_OPEN_ALWAYS | base::File::FLAG_ASYNC);
    EXPECT_TRUE(file.IsValid());
    EXPECT_TRUE(file.async());
  }

  {
    File file(file_path, base::File::FLAG_OPEN_ALWAYS);
    EXPECT_TRUE(file.IsValid());
    EXPECT_FALSE(file.async());
  }
}

TEST(FileTest, DeleteOpenFile) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("create_file_1");

  // Create a file.
  File file(file_path, base::File::FLAG_OPEN_ALWAYS | base::File::FLAG_READ |
                           base::File::FLAG_WIN_SHARE_DELETE);
  EXPECT_TRUE(file.IsValid());
  EXPECT_TRUE(file.created());
  EXPECT_EQ(base::File::FILE_OK, file.error_details());

  // Open an existing file and mark it as delete on close.
  File same_file(file_path,
                 base::File::FLAG_OPEN | base::File::FLAG_DELETE_ON_CLOSE |
                     base::File::FLAG_READ);
  EXPECT_TRUE(file.IsValid());
  EXPECT_FALSE(same_file.created());
  EXPECT_EQ(base::File::FILE_OK, same_file.error_details());

  // Close both handles and check that the file is gone.
  file.Close();
  same_file.Close();
  EXPECT_FALSE(base::PathExists(file_path));
}

TEST(FileTest, ReadWrite) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("read_write_file");
  File file(file_path,
            base::File::FLAG_CREATE | base::File::FLAG_READ |
                base::File::FLAG_WRITE);
  ASSERT_TRUE(file.IsValid());

  char data_to_write[] = "test";
  const int kTestDataSize = 4;

  // Write 0 bytes to the file.
  int bytes_written = file.Write(0, data_to_write, 0);
  EXPECT_EQ(0, bytes_written);

  // Write 0 bytes, with buf=nullptr.
  bytes_written = file.Write(0, nullptr, 0);
  EXPECT_EQ(0, bytes_written);

  // Write "test" to the file.
  bytes_written = file.Write(0, data_to_write, kTestDataSize);
  EXPECT_EQ(kTestDataSize, bytes_written);

  // Read from EOF.
  char data_read_1[32];
  int bytes_read = file.Read(kTestDataSize, data_read_1, kTestDataSize);
  EXPECT_EQ(0, bytes_read);

  // Read from somewhere in the middle of the file.
  const int kPartialReadOffset = 1;
  bytes_read = file.Read(kPartialReadOffset, data_read_1, kTestDataSize);
  EXPECT_EQ(kTestDataSize - kPartialReadOffset, bytes_read);
  for (int i = 0; i < bytes_read; i++)
    EXPECT_EQ(data_to_write[i + kPartialReadOffset], data_read_1[i]);

  // Read 0 bytes.
  bytes_read = file.Read(0, data_read_1, 0);
  EXPECT_EQ(0, bytes_read);

  // Read the entire file.
  bytes_read = file.Read(0, data_read_1, kTestDataSize);
  EXPECT_EQ(kTestDataSize, bytes_read);
  for (int i = 0; i < bytes_read; i++)
    EXPECT_EQ(data_to_write[i], data_read_1[i]);

  // Read again, but using the trivial native wrapper.
  bytes_read = file.ReadNoBestEffort(0, data_read_1, kTestDataSize);
  EXPECT_LE(bytes_read, kTestDataSize);
  for (int i = 0; i < bytes_read; i++)
    EXPECT_EQ(data_to_write[i], data_read_1[i]);

  // Write past the end of the file.
  const int kOffsetBeyondEndOfFile = 10;
  const int kPartialWriteLength = 2;
  bytes_written = file.Write(kOffsetBeyondEndOfFile,
                             data_to_write, kPartialWriteLength);
  EXPECT_EQ(kPartialWriteLength, bytes_written);

  // Make sure the file was extended.
  int64_t file_size = 0;
  EXPECT_TRUE(GetFileSize(file_path, &file_size));
  EXPECT_EQ(kOffsetBeyondEndOfFile + kPartialWriteLength, file_size);

  // Make sure the file was zero-padded.
  char data_read_2[32];
  bytes_read = file.Read(0, data_read_2, static_cast<int>(file_size));
  EXPECT_EQ(file_size, bytes_read);
  for (int i = 0; i < kTestDataSize; i++)
    EXPECT_EQ(data_to_write[i], data_read_2[i]);
  for (int i = kTestDataSize; i < kOffsetBeyondEndOfFile; i++)
    EXPECT_EQ(0, data_read_2[i]);
  for (int i = kOffsetBeyondEndOfFile; i < file_size; i++)
    EXPECT_EQ(data_to_write[i - kOffsetBeyondEndOfFile], data_read_2[i]);
}

TEST(FileTest, GetLastFileError) {
#if BUILDFLAG(IS_WIN)
  ::SetLastError(ERROR_ACCESS_DENIED);
#else
  errno = EACCES;
#endif
  EXPECT_EQ(File::FILE_ERROR_ACCESS_DENIED, File::GetLastFileError());

  base::ScopedTempDir temp_dir;
  EXPECT_TRUE(temp_dir.CreateUniqueTempDir());

  FilePath nonexistent_path(temp_dir.GetPath().Append("nonexistent"));
  File file(nonexistent_path, File::FLAG_OPEN | File::FLAG_READ);
  File::Error last_error = File::GetLastFileError();
  EXPECT_FALSE(file.IsValid());
  EXPECT_EQ(File::FILE_ERROR_NOT_FOUND, file.error_details());
  EXPECT_EQ(File::FILE_ERROR_NOT_FOUND, last_error);
}

TEST(FileTest, Append) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("append_file");
  File file(file_path, base::File::FLAG_CREATE | base::File::FLAG_APPEND);
  ASSERT_TRUE(file.IsValid());

  char data_to_write[] = "test";
  const int kTestDataSize = 4;

  // Write 0 bytes to the file.
  int bytes_written = file.Write(0, data_to_write, 0);
  EXPECT_EQ(0, bytes_written);

  // Write 0 bytes, with buf=nullptr.
  bytes_written = file.Write(0, nullptr, 0);
  EXPECT_EQ(0, bytes_written);

  // Write "test" to the file.
  bytes_written = file.Write(0, data_to_write, kTestDataSize);
  EXPECT_EQ(kTestDataSize, bytes_written);

  file.Close();
  File file2(file_path,
             base::File::FLAG_OPEN | base::File::FLAG_READ |
                 base::File::FLAG_APPEND);
  ASSERT_TRUE(file2.IsValid());

  // Test passing the file around.
  file = std::move(file2);
  EXPECT_FALSE(file2.IsValid());
  ASSERT_TRUE(file.IsValid());

  char append_data_to_write[] = "78";
  const int kAppendDataSize = 2;

  // Append "78" to the file.
  bytes_written = file.Write(0, append_data_to_write, kAppendDataSize);
  EXPECT_EQ(kAppendDataSize, bytes_written);

  // Read the entire file.
  char data_read_1[32];
  int bytes_read = file.Read(0, data_read_1,
                             kTestDataSize + kAppendDataSize);
  EXPECT_EQ(kTestDataSize + kAppendDataSize, bytes_read);
  for (int i = 0; i < kTestDataSize; i++)
    EXPECT_EQ(data_to_write[i], data_read_1[i]);
  for (int i = 0; i < kAppendDataSize; i++)
    EXPECT_EQ(append_data_to_write[i], data_read_1[kTestDataSize + i]);
}


TEST(FileTest, Length) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("truncate_file");
  File file(file_path,
            base::File::FLAG_CREATE | base::File::FLAG_READ |
                base::File::FLAG_WRITE);
  ASSERT_TRUE(file.IsValid());
  EXPECT_EQ(0, file.GetLength());

  // Write "test" to the file.
  char data_to_write[] = "test";
  int kTestDataSize = 4;
  int bytes_written = file.Write(0, data_to_write, kTestDataSize);
  EXPECT_EQ(kTestDataSize, bytes_written);

  // Extend the file.
  const int kExtendedFileLength = 10;
  int64_t file_size = 0;
  EXPECT_TRUE(file.SetLength(kExtendedFileLength));
  EXPECT_EQ(kExtendedFileLength, file.GetLength());
  EXPECT_TRUE(GetFileSize(file_path, &file_size));
  EXPECT_EQ(kExtendedFileLength, file_size);

  // Make sure the file was zero-padded.
  char data_read[32];
  int bytes_read = file.Read(0, data_read, static_cast<int>(file_size));
  EXPECT_EQ(file_size, bytes_read);
  for (int i = 0; i < kTestDataSize; i++)
    EXPECT_EQ(data_to_write[i], data_read[i]);
  for (int i = kTestDataSize; i < file_size; i++)
    EXPECT_EQ(0, data_read[i]);

  // Truncate the file.
  const int kTruncatedFileLength = 2;
  EXPECT_TRUE(file.SetLength(kTruncatedFileLength));
  EXPECT_EQ(kTruncatedFileLength, file.GetLength());
  EXPECT_TRUE(GetFileSize(file_path, &file_size));
  EXPECT_EQ(kTruncatedFileLength, file_size);

  // Make sure the file was truncated.
  bytes_read = file.Read(0, data_read, kTestDataSize);
  EXPECT_EQ(file_size, bytes_read);
  for (int i = 0; i < file_size; i++)
    EXPECT_EQ(data_to_write[i], data_read[i]);

#if !BUILDFLAG(IS_FUCHSIA)  // Fuchsia doesn't seem to support big files.
  // Expand the file past the 4 GB limit.
  const int64_t kBigFileLength = 5'000'000'000;
  EXPECT_TRUE(file.SetLength(kBigFileLength));
  EXPECT_EQ(kBigFileLength, file.GetLength());
  EXPECT_TRUE(GetFileSize(file_path, &file_size));
  EXPECT_EQ(kBigFileLength, file_size);
#endif

  // Close the file and reopen with base::File::FLAG_CREATE_ALWAYS, and make
  // sure the file is empty (old file was overridden).
  file.Close();
  file.Initialize(file_path,
                  base::File::FLAG_CREATE_ALWAYS | base::File::FLAG_WRITE);
  EXPECT_EQ(0, file.GetLength());
}

// Flakily fails: http://crbug.com/86494
#if BUILDFLAG(IS_ANDROID)
TEST(FileTest, TouchGetInfo) {
#else
TEST(FileTest, DISABLED_TouchGetInfo) {
#endif
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  File file(temp_dir.GetPath().Append("touch_get_info_file"),
            base::File::FLAG_CREATE | base::File::FLAG_WRITE |
                base::File::FLAG_WRITE_ATTRIBUTES);
  ASSERT_TRUE(file.IsValid());

  // Get info for a newly created file.
  base::File::Info info;
  EXPECT_TRUE(file.GetInfo(&info));

  // Add 2 seconds to account for possible rounding errors on
  // filesystems that use a 1s or 2s timestamp granularity.
  base::Time now = base::Time::Now() + base::Seconds(2);
  EXPECT_EQ(0, info.size);
  EXPECT_FALSE(info.is_directory);
  EXPECT_FALSE(info.is_symbolic_link);
  EXPECT_LE(info.last_accessed.ToDeltaSinceUnixEpoch(), now.ToDeltaSinceUnixEpoch());
  EXPECT_LE(info.last_modified.ToDeltaSinceUnixEpoch(), now.ToDeltaSinceUnixEpoch());
  EXPECT_LE(info.creation_time.ToDeltaSinceUnixEpoch(), now.ToDeltaSinceUnixEpoch());
  base::Time creation_time = info.creation_time;

  // Write "test" to the file.
  char data[] = "test";
  const int kTestDataSize = 4;
  int bytes_written = file.Write(0, data, kTestDataSize);
  EXPECT_EQ(kTestDataSize, bytes_written);

  // Change the last_accessed and last_modified dates.
  // It's best to add values that are multiples of 2 (in seconds)
  // to the current last_accessed and last_modified times, because
  // FATxx uses a 2s timestamp granularity.
  base::Time new_last_accessed = info.last_accessed + base::Seconds(234);
  base::Time new_last_modified = info.last_modified + base::Minutes(567);

  EXPECT_TRUE(file.SetTimes(new_last_accessed, new_last_modified));

  // Make sure the file info was updated accordingly.
  EXPECT_TRUE(file.GetInfo(&info));
  EXPECT_EQ(info.size, kTestDataSize);
  EXPECT_FALSE(info.is_directory);
  EXPECT_FALSE(info.is_symbolic_link);

  // ext2/ext3 and HPS/HPS+ seem to have a timestamp granularity of 1s.
#if BUILDFLAG(IS_POSIX)
  EXPECT_EQ(info.last_accessed.ToTimeVal().tv_sec,
            new_last_accessed.ToTimeVal().tv_sec);
  EXPECT_EQ(info.last_modified.ToTimeVal().tv_sec,
            new_last_modified.ToTimeVal().tv_sec);
#else
  EXPECT_EQ(info.last_accessed.ToDeltaSinceUnixEpoch(),
            new_last_accessed.ToDeltaSinceUnixEpoch());
  EXPECT_EQ(info.last_modified.ToDeltaSinceUnixEpoch(),
            new_last_modified.ToDeltaSinceUnixEpoch());
#endif

  EXPECT_EQ(info.creation_time.ToDeltaSinceUnixEpoch(),
            creation_time.ToDeltaSinceUnixEpoch());
}

// Test we can retrieve the file's creation time through File::GetInfo().
TEST(FileTest, GetInfoForCreationTime) {
  int64_t before_creation_time_s =
      base::Time::Now().ToDeltaSinceUnixEpoch().InSeconds();

  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("test_file");
  File file(file_path, base::File::FLAG_CREATE | base::File::FLAG_READ |
                           base::File::FLAG_WRITE);
  EXPECT_TRUE(file.IsValid());

  int64_t after_creation_time_s =
      base::Time::Now().ToDeltaSinceUnixEpoch().InSeconds();

  base::File::Info info;
  EXPECT_TRUE(file.GetInfo(&info));
  EXPECT_GE(info.creation_time.ToDeltaSinceUnixEpoch().InSeconds(),
            before_creation_time_s);
  EXPECT_LE(info.creation_time.ToDeltaSinceUnixEpoch().InSeconds(),
            after_creation_time_s);
}

TEST(FileTest, ReadAtCurrentPosition) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path =
      temp_dir.GetPath().Append("read_at_current_position");
  File file(file_path,
            base::File::FLAG_CREATE | base::File::FLAG_READ |
                base::File::FLAG_WRITE);
  EXPECT_TRUE(file.IsValid());

  const char kData[] = "test";
  const int kDataSize = sizeof(kData) - 1;
  EXPECT_EQ(kDataSize, file.Write(0, kData, kDataSize));

  EXPECT_EQ(0, file.Seek(base::File::FROM_BEGIN, 0));

  char buffer[kDataSize];
  int first_chunk_size = kDataSize / 2;
  EXPECT_EQ(first_chunk_size, file.ReadAtCurrentPos(buffer, first_chunk_size));
  EXPECT_EQ(kDataSize - first_chunk_size,
            file.ReadAtCurrentPos(buffer + first_chunk_size,
                                  kDataSize - first_chunk_size));
  EXPECT_EQ(std::string(buffer, buffer + kDataSize), std::string(kData));
}

TEST(FileTest, WriteAtCurrentPosition) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path =
      temp_dir.GetPath().Append("write_at_current_position");
  File file(file_path,
            base::File::FLAG_CREATE | base::File::FLAG_READ |
                base::File::FLAG_WRITE);
  EXPECT_TRUE(file.IsValid());

  const char kData[] = "test";
  const int kDataSize = sizeof(kData) - 1;

  int first_chunk_size = kDataSize / 2;
  EXPECT_EQ(first_chunk_size, file.WriteAtCurrentPos(kData, first_chunk_size));
  EXPECT_EQ(kDataSize - first_chunk_size,
            file.WriteAtCurrentPos(kData + first_chunk_size,
                                   kDataSize - first_chunk_size));

  char buffer[kDataSize];
  EXPECT_EQ(kDataSize, file.Read(0, buffer, kDataSize));
  EXPECT_EQ(std::string(buffer, buffer + kDataSize), std::string(kData));
}

TEST(FileTest, Seek) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("seek_file");
  File file(file_path,
            base::File::FLAG_CREATE | base::File::FLAG_READ |
                base::File::FLAG_WRITE);
  ASSERT_TRUE(file.IsValid());

  const int64_t kOffset = 10;
  EXPECT_EQ(kOffset, file.Seek(base::File::FROM_BEGIN, kOffset));
  EXPECT_EQ(2 * kOffset, file.Seek(base::File::FROM_CURRENT, kOffset));
  EXPECT_EQ(kOffset, file.Seek(base::File::FROM_CURRENT, -kOffset));
  EXPECT_TRUE(file.SetLength(kOffset * 2));
  EXPECT_EQ(kOffset, file.Seek(base::File::FROM_END, -kOffset));
}

TEST(FileTest, Duplicate) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");
  File file(file_path,(base::File::FLAG_CREATE |
                       base::File::FLAG_READ |
                       base::File::FLAG_WRITE));
  ASSERT_TRUE(file.IsValid());

  File file2(file.Duplicate());
  ASSERT_TRUE(file2.IsValid());

  // Write through one handle, close it, read through the other.
  static const char kData[] = "now is a good time.";
  static const int kDataLen = sizeof(kData) - 1;

  ASSERT_EQ(0, file.Seek(base::File::FROM_CURRENT, 0));
  ASSERT_EQ(0, file2.Seek(base::File::FROM_CURRENT, 0));
  ASSERT_EQ(kDataLen, file.WriteAtCurrentPos(kData, kDataLen));
  ASSERT_EQ(kDataLen, file.Seek(base::File::FROM_CURRENT, 0));
  ASSERT_EQ(kDataLen, file2.Seek(base::File::FROM_CURRENT, 0));
  file.Close();
  char buf[kDataLen];
  ASSERT_EQ(kDataLen, file2.Read(0, &buf[0], kDataLen));
  ASSERT_EQ(std::string(kData, kDataLen), std::string(&buf[0], kDataLen));
}

TEST(FileTest, DuplicateDeleteOnClose) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");
  File file(file_path,(base::File::FLAG_CREATE |
                       base::File::FLAG_READ |
                       base::File::FLAG_WRITE |
                       base::File::FLAG_DELETE_ON_CLOSE));
  ASSERT_TRUE(file.IsValid());
  File file2(file.Duplicate());
  ASSERT_TRUE(file2.IsValid());
  file.Close();
  file2.Close();
  ASSERT_FALSE(base::PathExists(file_path));
}

// TODO(chokobole):
// #if BUILDFLAG(ENABLE_BASE_TRACING)
// TEST(FileTest, TracedValueSupport) {
//   base::ScopedTempDir temp_dir;
//   ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
//   FilePath file_path = temp_dir.GetPath().Append("file");

//   File file(file_path,
//             (base::File::FLAG_CREATE | base::File::FLAG_READ |
//              base::File::FLAG_WRITE | base::File::FLAG_DELETE_ON_CLOSE));
//   ASSERT_TRUE(file.IsValid());

//   EXPECT_EQ(perfetto::TracedValueToString(file),
//             "{is_valid:true,created:true,async:false,error_details:FILE_OK}");
// }
// #endif  // BUILDFLAG(ENABLE_BASE_TRACING)

#if BUILDFLAG(IS_WIN)
// Flakily times out on Windows, see http://crbug.com/846276.
#define MAYBE_WriteDataToLargeOffset DISABLED_WriteDataToLargeOffset
#else
#define MAYBE_WriteDataToLargeOffset WriteDataToLargeOffset
#endif
TEST(FileTest, MAYBE_WriteDataToLargeOffset) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");
  File file(file_path,
            (base::File::FLAG_CREATE | base::File::FLAG_READ |
             base::File::FLAG_WRITE | base::File::FLAG_DELETE_ON_CLOSE));
  ASSERT_TRUE(file.IsValid());

  const char kData[] = "this file is sparse.";
  const int kDataLen = sizeof(kData) - 1;
  const int64_t kLargeFileOffset = (1LL << 31);

  // If the file fails to write, it is probably we are running out of disk space
  // and the file system doesn't support sparse file.
  if (file.Write(kLargeFileOffset - kDataLen - 1, kData, kDataLen) < 0)
    return;

  ASSERT_EQ(kDataLen, file.Write(kLargeFileOffset + 1, kData, kDataLen));
}

TEST(FileTest, AddFlagsForPassingToUntrustedProcess) {
  {
    uint32_t flags = base::File::FLAG_OPEN | base::File::FLAG_READ;
    flags = base::File::AddFlagsForPassingToUntrustedProcess(flags);
    EXPECT_EQ(flags, base::File::FLAG_OPEN | base::File::FLAG_READ);
  }
  {
    uint32_t flags = base::File::FLAG_OPEN | base::File::FLAG_WRITE;
    flags = base::File::AddFlagsForPassingToUntrustedProcess(flags);
    EXPECT_EQ(flags, base::File::FLAG_OPEN | base::File::FLAG_WRITE |
                         base::File::FLAG_WIN_NO_EXECUTE);
  }
}

#if BUILDFLAG(IS_WIN)
TEST(FileTest, GetInfoForDirectory) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath empty_dir =
      temp_dir.GetPath().Append(FILE_PATH_LITERAL("gpfi_test"));
  ASSERT_TRUE(CreateDirectory(empty_dir));

  base::File dir(
      ::CreateFile(empty_dir.value().c_str(), GENERIC_READ | GENERIC_WRITE,
                   FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                   OPEN_EXISTING,
                   FILE_FLAG_BACKUP_SEMANTICS,  // Needed to open a directory.
                   NULL));
  ASSERT_TRUE(dir.IsValid());

  base::File::Info info;
  EXPECT_TRUE(dir.GetInfo(&info));
  EXPECT_TRUE(info.is_directory);
  EXPECT_FALSE(info.is_symbolic_link);
  EXPECT_EQ(0, info.size);
}

TEST(FileTest, DeleteNoop) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");

  // Creating and closing a file with DELETE perms should do nothing special.
  File file(file_path,
            (base::File::FLAG_CREATE | base::File::FLAG_READ |
             base::File::FLAG_WRITE | base::File::FLAG_CAN_DELETE_ON_CLOSE));
  ASSERT_TRUE(file.IsValid());
  file.Close();
  ASSERT_TRUE(base::PathExists(file_path));
}

TEST(FileTest, Delete) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");

  // Creating a file with DELETE and then marking for delete on close should
  // delete it.
  File file(file_path,
            (base::File::FLAG_CREATE | base::File::FLAG_READ |
             base::File::FLAG_WRITE | base::File::FLAG_CAN_DELETE_ON_CLOSE));
  ASSERT_TRUE(file.IsValid());
  ASSERT_TRUE(file.DeleteOnClose(true));
  file.Close();
  ASSERT_FALSE(base::PathExists(file_path));
}

TEST(FileTest, DeleteThenRevoke) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");

  // Creating a file with DELETE, marking it for delete, then clearing delete on
  // close should not delete it.
  File file(file_path,
            (base::File::FLAG_CREATE | base::File::FLAG_READ |
             base::File::FLAG_WRITE | base::File::FLAG_CAN_DELETE_ON_CLOSE));
  ASSERT_TRUE(file.IsValid());
  ASSERT_TRUE(file.DeleteOnClose(true));
  ASSERT_TRUE(file.DeleteOnClose(false));
  file.Close();
  ASSERT_TRUE(base::PathExists(file_path));
}

TEST(FileTest, IrrevokableDeleteOnClose) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");

  // DELETE_ON_CLOSE cannot be revoked by this opener.
  File file(file_path,
            (base::File::FLAG_CREATE | base::File::FLAG_READ |
             base::File::FLAG_WRITE | base::File::FLAG_DELETE_ON_CLOSE |
             base::File::FLAG_WIN_SHARE_DELETE |
             base::File::FLAG_CAN_DELETE_ON_CLOSE));
  ASSERT_TRUE(file.IsValid());
  // https://msdn.microsoft.com/library/windows/desktop/aa364221.aspx says that
  // setting the dispositon has no effect if the handle was opened with
  // FLAG_DELETE_ON_CLOSE. Do not make the test's success dependent on whether
  // or not SetFileInformationByHandle indicates success or failure. (It happens
  // to indicate success on Windows 10.)
  file.DeleteOnClose(false);
  file.Close();
  ASSERT_FALSE(base::PathExists(file_path));
}

TEST(FileTest, IrrevokableDeleteOnCloseOther) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");

  // DELETE_ON_CLOSE cannot be revoked by another opener.
  File file(file_path,
            (base::File::FLAG_CREATE | base::File::FLAG_READ |
             base::File::FLAG_WRITE | base::File::FLAG_DELETE_ON_CLOSE |
             base::File::FLAG_WIN_SHARE_DELETE |
             base::File::FLAG_CAN_DELETE_ON_CLOSE));
  ASSERT_TRUE(file.IsValid());

  File file2(file_path,
             (base::File::FLAG_OPEN | base::File::FLAG_READ |
              base::File::FLAG_WRITE | base::File::FLAG_WIN_SHARE_DELETE |
              base::File::FLAG_CAN_DELETE_ON_CLOSE));
  ASSERT_TRUE(file2.IsValid());

  file2.DeleteOnClose(false);
  file2.Close();
  ASSERT_TRUE(base::PathExists(file_path));
  file.Close();
  ASSERT_FALSE(base::PathExists(file_path));
}

TEST(FileTest, DeleteWithoutPermission) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");

  // It should not be possible to mark a file for deletion when it was not
  // created/opened with DELETE.
  File file(file_path, (base::File::FLAG_CREATE | base::File::FLAG_READ |
                        base::File::FLAG_WRITE));
  ASSERT_TRUE(file.IsValid());
  ASSERT_FALSE(file.DeleteOnClose(true));
  file.Close();
  ASSERT_TRUE(base::PathExists(file_path));
}

TEST(FileTest, UnsharedDeleteOnClose) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");

  // Opening with DELETE_ON_CLOSE when a previous opener hasn't enabled sharing
  // will fail.
  File file(file_path, (base::File::FLAG_CREATE | base::File::FLAG_READ |
                        base::File::FLAG_WRITE));
  ASSERT_TRUE(file.IsValid());
  File file2(
      file_path,
      (base::File::FLAG_OPEN | base::File::FLAG_READ | base::File::FLAG_WRITE |
       base::File::FLAG_DELETE_ON_CLOSE | base::File::FLAG_WIN_SHARE_DELETE));
  ASSERT_FALSE(file2.IsValid());

  file.Close();
  ASSERT_TRUE(base::PathExists(file_path));
}

TEST(FileTest, NoDeleteOnCloseWithMappedFile) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");

  // Mapping a file into memory blocks DeleteOnClose.
  File file(file_path,
            (base::File::FLAG_CREATE | base::File::FLAG_READ |
             base::File::FLAG_WRITE | base::File::FLAG_CAN_DELETE_ON_CLOSE));
  ASSERT_TRUE(file.IsValid());
  ASSERT_EQ(5, file.WriteAtCurrentPos("12345", 5));

  {
    base::MemoryMappedFile mapping;
    ASSERT_TRUE(mapping.Initialize(file.Duplicate()));
    ASSERT_EQ(5U, mapping.length());

    EXPECT_FALSE(file.DeleteOnClose(true));
  }

  file.Close();
  ASSERT_TRUE(base::PathExists(file_path));
}

// Check that we handle the async bit being set incorrectly in a sane way.
TEST(FileTest, UseSyncApiWithAsyncFile) {
  base::ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateUniqueTempDir());
  FilePath file_path = temp_dir.GetPath().Append("file");

  File file(file_path, base::File::FLAG_CREATE | base::File::FLAG_WRITE |
                           base::File::FLAG_ASYNC);
  File lying_file(file.TakePlatformFile(), false /* async */);
  ASSERT_TRUE(lying_file.IsValid());

  ASSERT_EQ(lying_file.WriteAtCurrentPos("12345", 5), -1);
}

TEST(FileDeathTest, InvalidFlags) {
  EXPECT_CHECK_DEATH_WITH(
      {
        // When this test is running as Admin, TMP gets ignored and temporary
        // files/folders are created in %ProgramFiles%. This means that the
        // temporary folder created by the death test never gets deleted, as it
        // crashes before the `base::ScopedTempDir` goes out of scope and also
        // does not get automatically cleaned by by the test runner.
        //
        // To avoid this from happening, this death test explicitly creates the
        // temporary folder in TMP, which is set by the test runner parent
        // process to a temporary folder for the test. This means that the
        // folder created here is always deleted during test runner cleanup.
        std::string tmp_folder;
        ASSERT_TRUE(base::Environment::Create()->GetVar("TMP", &tmp_folder));
        base::ScopedTempDir temp_dir;
        ASSERT_TRUE(temp_dir.CreateUniqueTempDirUnderPath(
            base::FilePath(base::UTF8ToWide(tmp_folder))));
        FilePath file_path = temp_dir.GetPath().Append("file");

        File file(file_path,
                  base::File::FLAG_CREATE | base::File::FLAG_WIN_EXECUTE |
                      base::File::FLAG_READ | base::File::FLAG_WIN_NO_EXECUTE);
      },
      "FLAG_WIN_NO_EXECUTE");
}
#endif  // BUILDFLAG(IS_WIN)

}  // namespace tachyon
