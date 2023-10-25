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

  if constexpr (std::is_same_v<ColumnTy, AdviceColumn>) {
    AnyColumn any(ColumnTy(1, kSecondPhase));
    EXPECT_EQ(any.phase(), kSecondPhase);
    AnyColumn any2;
    EXPECT_EQ(any2.phase(), kFirstPhase);
    any2 = ColumnTy(1, kSecondPhase);
    EXPECT_EQ(any2.phase(), kSecondPhase);
  }
}

TYPED_TEST(ColumnTest, NonAnyColumnConstruction) {
  using ColumnTy = TypeParam;

  ColumnTy c(ColumnTy(1));
  EXPECT_EQ(c.type(), ColumnTy::kDefaultType);
  c = ColumnTy(1);
  EXPECT_EQ(c.type(), ColumnTy::kDefaultType);
  ColumnTy c2(AnyColumn(1));
  EXPECT_EQ(c2.type(), ColumnTy::kDefaultType);
  ColumnTy c3;
  EXPECT_EQ(c3.type(), ColumnTy::kDefaultType);
  c3 = AnyColumn(1);
  EXPECT_EQ(c3.type(), ColumnTy::kDefaultType);
}

}  // namespace tachyon::zk
