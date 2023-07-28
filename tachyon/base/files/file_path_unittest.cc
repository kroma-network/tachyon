#include "tachyon/base/files/file_path.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace tachyon::base {

TEST(FilePathTest, GetComponents) {
  std::vector<std::string> components;
  FilePath("foo/bar").GetComponents(&components);
  EXPECT_THAT(components,
              testing::ContainerEq(std::vector<std::string>{"foo", "bar"}));
  FilePath("/foo/bar").GetComponents(&components);
  EXPECT_THAT(components, testing::ContainerEq(
                              std::vector<std::string>{"/", "foo", "bar"}));
  FilePath("./foo/bar").GetComponents(&components);
  EXPECT_THAT(components, testing::ContainerEq(
                              std::vector<std::string>{".", "foo", "bar"}));
}

TEST(FilePathTest, IsParent) {
  FilePath path("foo/bar");
  EXPECT_TRUE(path.IsParent(FilePath("foo/bar/baz.jpg")));
  EXPECT_FALSE(path.IsParent(FilePath("foo/bar")));
  EXPECT_FALSE(path.IsParent(FilePath("abc/foo/bar/baz.jpg")));
}

TEST(FilePathTest, AppendRelativePath) {
  FilePath path("foo/bar");
  FilePath path_appended;
  EXPECT_TRUE(
      path.AppendRelativePath(FilePath("foo/bar/baz.jpg"), &path_appended));
  EXPECT_EQ(FilePath("baz.jpg"), path_appended);
}

TEST(FilePathTest, DirName) {
  FilePath path("foo/bar");
  EXPECT_EQ(path.DirName(), FilePath("foo"));
  EXPECT_EQ(path.DirName().DirName(), FilePath(""));
  EXPECT_EQ(FilePath("/").DirName(), FilePath("/"));
}

TEST(FilePathTest, BaseName) {
  FilePath path("foo/bar");
  EXPECT_EQ(path.BaseName(), FilePath("bar"));
  EXPECT_EQ(path.DirName().BaseName(), FilePath("foo"));
  EXPECT_EQ(FilePath(""), FilePath(""));
  EXPECT_EQ(FilePath("/").BaseName(), FilePath(""));
}

TEST(FilePathTest, Extension) {
  EXPECT_EQ(FilePath("foo/bar.jpg").Extension(), ".jpg");
}

TEST(FilePathTest, IsAbsolute) {
  EXPECT_FALSE(FilePath("foo/bar").IsAbsolute());
  EXPECT_TRUE(FilePath("/foo/bar").IsAbsolute());
}

TEST(FilePathTest, EndsWithSeparator) {
  EXPECT_TRUE(FilePath("foo/").EndsWithSeparator());
  EXPECT_FALSE(FilePath("foo").EndsWithSeparator());
}

TEST(FilePathTest, AsEndingWithSeparator) {
  EXPECT_EQ(FilePath("foo/").AsEndingWithSeparator(), FilePath("foo/"));
  EXPECT_EQ(FilePath("foo").AsEndingWithSeparator(), FilePath("foo/"));
}

TEST(FilePathTest, StripTrailingSeparators) {
  EXPECT_EQ(FilePath("foo/").StripTrailingSeparators(), FilePath("foo"));
  EXPECT_EQ(FilePath("foo//").StripTrailingSeparators(), FilePath("foo"));
  EXPECT_EQ(FilePath("foo").StripTrailingSeparators(), FilePath("foo"));
  EXPECT_EQ(FilePath("/").StripTrailingSeparators(), FilePath("/"));
}

TEST(FilePathTest, AppendComponent) {
  {
    FilePath path("foo/bar");
    EXPECT_EQ(path.Append("baz"), FilePath("foo/bar/baz"));
  }
  {
    FilePath path;
    EXPECT_EQ(path.Append("foo"), FilePath("foo"));
  }
  {
    FilePath path(".");
    EXPECT_EQ(path.Append("foo"), FilePath("./foo"));
  }
}

TEST(FilePathTest, ReferencesParent) {
  EXPECT_TRUE(FilePath("..").ReferencesParent());
  EXPECT_TRUE(FilePath("../foo").ReferencesParent());
  EXPECT_TRUE(FilePath("foo/../bar").ReferencesParent());
  EXPECT_FALSE(FilePath(".").ReferencesParent());
  EXPECT_FALSE(FilePath("foo/bar").ReferencesParent());
}

}  // namespace tachyon::base
