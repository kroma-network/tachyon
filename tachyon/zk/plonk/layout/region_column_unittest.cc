#include "tachyon/zk/plonk/layout/region_column.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

namespace tachyon::zk::plonk {

TEST(RegionColumnTest, Order) {
  EXPECT_FALSE(RegionColumn(FixedColumnKey()) < RegionColumn(FixedColumnKey()));

  EXPECT_FALSE(RegionColumn(Selector::Simple(0)) <
               RegionColumn(Selector::Simple(0)));
  EXPECT_TRUE(RegionColumn(Selector::Simple(0)) <
              RegionColumn(Selector::Simple(1)));

  EXPECT_FALSE(RegionColumn(Selector::Complex(0)) <
               RegionColumn(Selector::Complex(0)));
  EXPECT_TRUE(RegionColumn(Selector::Complex(0)) <
              RegionColumn(Selector::Complex(1)));

  EXPECT_TRUE(RegionColumn(AnyColumnKey()) < RegionColumn(Selector::Simple(0)));
  EXPECT_TRUE(RegionColumn(AnyColumnKey()) <
              RegionColumn(Selector::Complex(0)));

  EXPECT_TRUE(RegionColumn(InstanceColumnKey()) <
              RegionColumn(Selector::Simple(0)));
  EXPECT_TRUE(RegionColumn(InstanceColumnKey()) <
              RegionColumn(Selector::Complex(0)));

  EXPECT_TRUE(RegionColumn(AdviceColumnKey()) <
              RegionColumn(Selector::Simple(0)));
  EXPECT_TRUE(RegionColumn(AdviceColumnKey()) <
              RegionColumn(Selector::Complex(0)));

  EXPECT_TRUE(RegionColumn(FixedColumnKey()) <
              RegionColumn(Selector::Simple(0)));
  EXPECT_TRUE(RegionColumn(FixedColumnKey()) <
              RegionColumn(Selector::Complex(0)));
}

TEST(RegionColumnTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(std::make_tuple(
      RegionColumn(FixedColumnKey(0)), RegionColumn(FixedColumnKey(1)),
      RegionColumn(AdviceColumnKey(0)), RegionColumn(AdviceColumnKey(1)),
      RegionColumn(InstanceColumnKey(0)), RegionColumn(InstanceColumnKey(1)),
      RegionColumn(Selector::Simple(0)), RegionColumn(Selector::Simple(1)),
      RegionColumn(Selector::Complex(0)), RegionColumn(Selector::Complex(1)))));
}

}  // namespace tachyon::zk::plonk
