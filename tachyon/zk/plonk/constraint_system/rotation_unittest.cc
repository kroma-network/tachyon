#include "tachyon/zk/plonk/constraint_system/rotation.h"

#include "gtest/gtest.h"

namespace tachyon::zk {

TEST(RotationTest, GetRotationIdx) {
  struct {
    int32_t idx;
    Rotation rotation;
    int32_t scale;
    int32_t size;
    RowIndex expected;
  } tests[] = {
      // (2 + 1 * 3) % 2 = 1
      {2, Rotation(1), 3, 2, 1},
      // (4 + 2 * 5) % 3 = 2
      {4, Rotation(2), 5, 3, 2},
      // (6 + 3 * 4) % 6 = 0
      {6, Rotation(3), 4, 6, 0},
  };
  for (const auto& test : tests) {
    EXPECT_EQ(test.expected,
              test.rotation.GetIndex(test.idx, test.scale, test.size));
  }
}

}  // namespace tachyon::zk
