#include "tachyon/version.h"

#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

namespace tachyon {

TEST(VersionTest, CompileTimeVersionTest) {
  EXPECT_EQ(absl::Substitute("$0.$1.$2", TACHYON_VERSION_MAJOR,
                             TACHYON_VERSION_MINOR, TACHYON_VERSION_PATCH),
            TACHYON_VERSION_STR);
  EXPECT_EQ(TACHYON_VERSION_MAJOR * 10000 + TACHYON_VERSION_MINOR * 100 +
                TACHYON_VERSION_PATCH,
            TACHYON_VERSION);
}

TEST(VersionTest, RunTimeVersionTest) {
  EXPECT_EQ(TACHYON_VERSION, GetRuntimeVersion());
  EXPECT_EQ(TACHYON_VERSION_STR, GetRuntimeVersionStr());
  EXPECT_EQ(TACHYON_VERSION_FULL_STR, GetRuntimeFullVersionStr());
}

}  // namespace tachyon
