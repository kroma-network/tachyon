#include "tachyon/zk/plonk/circuit/column.h"

#include "gtest/gtest.h"

namespace tachyon::zk {

template <typename ColumnType>
class ColumnTest : public testing::Test {};

using ColumnTypes = testing::Types<FixedColumn, AdviceColumn, InstanceColumn>;
TYPED_TEST_SUITE(ColumnTest, ColumnTypes);

TYPED_TEST(ColumnTest, AnyColumnConstruction) {
  using ColumnTy = TypeParam;

  AnyColumn any(ColumnTy(1));
  EXPECT_EQ(any.type(), ColumnTy::kDefaultType);
  AnyColumn any2;
  EXPECT_EQ(any2.type(), ColumnType::kAny);
  any2 = ColumnTy(1);
  EXPECT_EQ(any2.type(), ColumnTy::kDefaultType);
}

TYPED_TEST(ColumnTest, NonAnyColumnConstruction) {
  using ColumnTy = TypeParam;

  ColumnTy c(ColumnTy(1));
  EXPECT_EQ(c.type(), ColumnTy::kDefaultType);
  c = ColumnTy(1);
  EXPECT_EQ(c.type(), ColumnTy::kDefaultType);
}

}  // namespace tachyon::zk
