#include "tachyon/zk/plonk/circuit/column_key.h"

#include "absl/hash/hash_testing.h"
#include "gtest/gtest.h"

namespace tachyon::zk {

template <typename ColumnKey>
class ColumnKeyTest : public testing::Test {};

using ColumnKeyTypes =
    testing::Types<FixedColumnKey, AdviceColumnKey, InstanceColumnKey>;
TYPED_TEST_SUITE(ColumnKeyTest, ColumnKeyTypes);

TYPED_TEST(ColumnKeyTest, AnyColumnKeyConstruction) {
  using ColumnKey = TypeParam;

  AnyColumnKey any(ColumnKey(1));
  EXPECT_EQ(any.type(), ColumnKey::kDefaultType);
  AnyColumnKey any2;
  EXPECT_EQ(any2.type(), ColumnType::kAny);
  any2 = ColumnKey(1);
  EXPECT_EQ(any2.type(), ColumnKey::kDefaultType);

  if constexpr (std::is_same_v<ColumnKey, AdviceColumnKey>) {
    AnyColumnKey any(ColumnKey(1, kSecondPhase));
    EXPECT_EQ(any.phase(), kSecondPhase);
    AnyColumnKey any2;
    EXPECT_EQ(any2.phase(), kFirstPhase);
    any2 = ColumnKey(1, kSecondPhase);
    EXPECT_EQ(any2.phase(), kSecondPhase);
  }
}

TYPED_TEST(ColumnKeyTest, NonAnyColumnKeyConstruction) {
  using ColumnKey = TypeParam;

  ColumnKey c(ColumnKey(1));
  EXPECT_EQ(c.type(), ColumnKey::kDefaultType);
  c = ColumnKey(1);
  EXPECT_EQ(c.type(), ColumnKey::kDefaultType);
  ColumnKey c2(AnyColumnKey(1));
  EXPECT_EQ(c2.type(), ColumnKey::kDefaultType);
  ColumnKey c3;
  EXPECT_EQ(c3.type(), ColumnKey::kDefaultType);
  c3 = AnyColumnKey(1);
  EXPECT_EQ(c3.type(), ColumnKey::kDefaultType);
}

TYPED_TEST(ColumnKeyTest, Hash) {
  using ColumnKey = TypeParam;
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      std::make_tuple(ColumnKey(), ColumnKey(1))));
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
