#include "tachyon/zk/plonk/circuit/region_column.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

namespace tachyon::zk {

TEST(RegionColumnTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(std::make_tuple(
      RegionColumn(FixedColumnKey(0)), RegionColumn(FixedColumnKey(1)),
      RegionColumn(AdviceColumnKey(0)), RegionColumn(AdviceColumnKey(1)),
      RegionColumn(InstanceColumnKey(0)), RegionColumn(InstanceColumnKey(1)),
      RegionColumn(Selector::Simple(0)), RegionColumn(Selector::Simple(1)),
      RegionColumn(Selector::Complex(0)), RegionColumn(Selector::Complex(1)))));
}

}  // namespace tachyon::zk
