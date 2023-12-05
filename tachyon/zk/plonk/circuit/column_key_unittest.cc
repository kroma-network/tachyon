#include "tachyon/zk/plonk/circuit/column_key.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

namespace tachyon::zk {

template <typename ColumnKeyType>
class ColumnKeyTest : public testing::Test {};

using ColumnKeyTypes =
    testing::Types<FixedColumnKey, AdviceColumnKey, InstanceColumnKey>;
TYPED_TEST_SUITE(ColumnKeyTest, ColumnKeyTypes);

TYPED_TEST(ColumnKeyTest, AnyColumnKeyConstruction) {
  using ColumnKeyTy = TypeParam;

  AnyColumnKey any(ColumnKeyTy(1));
  EXPECT_EQ(any.type(), ColumnKeyTy::kDefaultType);
  AnyColumnKey any2;
  EXPECT_EQ(any2.type(), ColumnType::kAny);
  any2 = ColumnKeyTy(1);
  EXPECT_EQ(any2.type(), ColumnKeyTy::kDefaultType);

  if constexpr (std::is_same_v<ColumnKeyTy, AdviceColumnKey>) {
    AnyColumnKey any(ColumnKeyTy(1, kSecondPhase));
    EXPECT_EQ(any.phase(), kSecondPhase);
    AnyColumnKey any2;
    EXPECT_EQ(any2.phase(), kFirstPhase);
    any2 = ColumnKeyTy(1, kSecondPhase);
    EXPECT_EQ(any2.phase(), kSecondPhase);
  }
}

TYPED_TEST(ColumnKeyTest, NonAnyColumnKeyConstruction) {
  using ColumnKeyTy = TypeParam;

  ColumnKeyTy c(ColumnKeyTy(1));
  EXPECT_EQ(c.type(), ColumnKeyTy::kDefaultType);
  c = ColumnKeyTy(1);
  EXPECT_EQ(c.type(), ColumnKeyTy::kDefaultType);
  ColumnKeyTy c2(AnyColumnKey(1));
  EXPECT_EQ(c2.type(), ColumnKeyTy::kDefaultType);
  ColumnKeyTy c3;
  EXPECT_EQ(c3.type(), ColumnKeyTy::kDefaultType);
  c3 = AnyColumnKey(1);
  EXPECT_EQ(c3.type(), ColumnKeyTy::kDefaultType);
}

TYPED_TEST(ColumnKeyTest, Hash) {
  using ColumnKeyTy = TypeParam;
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(ColumnKeyTy(), ColumnKeyTy(1))));
}

TEST(ColumnKeyBaseTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(ColumnKeyBase(), ColumnKeyBase(ColumnType::kFixed, 0),
                      ColumnKeyBase(ColumnType::kFixed, 1),
                      ColumnKeyBase(ColumnType::kAdvice, 0),
                      ColumnKeyBase(ColumnType::kAdvice, 1),
                      ColumnKeyBase(ColumnType::kInstance, 0),
                      ColumnKeyBase(ColumnType::kInstance, 1))));
}

}  // namespace tachyon::zk
