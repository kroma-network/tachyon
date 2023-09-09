#include "tachyon/c/version.h"

#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

TEST(VersionTest, CompileTimeVersionTest) {
  EXPECT_EQ(absl::Substitute("$0.$1.$2", TACHYON_C_VERSION_MAJOR,
                             TACHYON_C_VERSION_MINOR, TACHYON_C_VERSION_PATCH),
            TACHYON_C_VERSION_STR);
  EXPECT_EQ(TACHYON_C_VERSION_MAJOR * 10000 + TACHYON_C_VERSION_MINOR * 100 +
                TACHYON_C_VERSION_PATCH,
            TACHYON_C_VERSION);
}

TEST(VersionTest, RunTimeVersionTest) {
  EXPECT_EQ(TACHYON_C_VERSION, tachyon_get_runtime_version());
  EXPECT_EQ(TACHYON_C_VERSION_STR,
            std::string_view(tachyon_get_runtime_version_str()));
  EXPECT_EQ(TACHYON_C_VERSION_FULL_STR,
            std::string_view(tachyon_get_runtime_full_version_str()));
}
