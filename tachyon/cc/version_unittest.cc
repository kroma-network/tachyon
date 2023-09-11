#include "tachyon/cc/version.h"

#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

namespace tachyon::cc {

TEST(VersionTest, CompileTimeVersionTest) {
  EXPECT_EQ(
      absl::Substitute("$0.$1.$2", TACHYON_CC_VERSION_MAJOR,
                       TACHYON_CC_VERSION_MINOR, TACHYON_CC_VERSION_PATCH),
      TACHYON_CC_VERSION_STR);
  EXPECT_EQ(TACHYON_CC_VERSION_MAJOR * 10000 + TACHYON_CC_VERSION_MINOR * 100 +
                TACHYON_CC_VERSION_PATCH,
            TACHYON_CC_VERSION);
}

TEST(VersionTest, RunTimeVersionTest) {
  EXPECT_EQ(TACHYON_CC_VERSION, GetRuntimeVersion());
  EXPECT_EQ(TACHYON_CC_VERSION_STR, GetRuntimeVersionStr());
  EXPECT_EQ(TACHYON_CC_VERSION_FULL_STR, GetRuntimeFullVersionStr());
}

}  // namespace tachyon::cc
