#include "tachyon/zk/plonk/constraint_system/constraint_system.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk::plonk {

namespace {

class ConstraintSystemTest : public math::FiniteFieldTest<math::GF7> {};

}  // namespace

TEST_F(ConstraintSystemTest, EnableConstant) {
  ConstraintSystem<math::GF7> constraint_system;
  std::vector<FixedColumnKey> expected_constants;
  std::vector<AnyColumnKey> expected_permutation_columns;
  EXPECT_EQ(constraint_system.constants(), expected_constants);
  EXPECT_EQ(constraint_system.permutation().columns(),
            expected_permutation_columns);

  FixedColumnKey column = constraint_system.CreateFixedColumn();
  constraint_system.EnableConstant(column);
  expected_constants.push_back(column);
  EXPECT_EQ(constraint_system.constants(), expected_constants);
  expected_permutation_columns.push_back(AnyColumnKey(column));
  EXPECT_EQ(constraint_system.permutation().columns(),
            expected_permutation_columns);

  constraint_system.EnableConstant(column);
  EXPECT_EQ(constraint_system.constants(), expected_constants);
  EXPECT_EQ(constraint_system.permutation().columns(),
            expected_permutation_columns);
}

// TODO(chokobole): Add tests for Lookup and LookupAny.

TEST_F(ConstraintSystemTest, QueryFixedIndex) {
  ConstraintSystem<math::GF7> constraint_system;
  FixedColumnKey column = constraint_system.CreateFixedColumn();
  Rotation rotation = Rotation::Cur();
  EXPECT_EQ(constraint_system.QueryFixedIndex(column, rotation), 0);
  EXPECT_EQ(constraint_system.QueryFixedIndex(column, rotation), 0);

  column = constraint_system.CreateFixedColumn();
  rotation = Rotation::Cur();
  EXPECT_EQ(constraint_system.QueryFixedIndex(column, rotation), 1);

  rotation = Rotation::Next();
  EXPECT_EQ(constraint_system.QueryFixedIndex(column, rotation), 2);
}

TEST_F(ConstraintSystemTest, QueryAdviceIndex) {
  ConstraintSystem<math::GF7> constraint_system;
  AdviceColumnKey column = constraint_system.CreateAdviceColumn();
  Rotation rotation = Rotation::Cur();

  EXPECT_EQ(constraint_system.QueryAdviceIndex(column, rotation), 0);
  EXPECT_EQ(constraint_system.QueryAdviceIndex(column, rotation), 0);

  column = constraint_system.CreateAdviceColumn();
  rotation = Rotation::Cur();
  EXPECT_EQ(constraint_system.QueryAdviceIndex(column, rotation), 1);

  rotation = Rotation::Next();
  EXPECT_EQ(constraint_system.QueryAdviceIndex(column, rotation), 2);
}

TEST_F(ConstraintSystemTest, QueryInstanceIndex) {
  ConstraintSystem<math::GF7> constraint_system;
  InstanceColumnKey column = constraint_system.CreateInstanceColumn();
  Rotation rotation = Rotation::Cur();
  EXPECT_EQ(constraint_system.QueryInstanceIndex(column, rotation), 0);
  EXPECT_EQ(constraint_system.QueryInstanceIndex(column, rotation), 0);

  column = constraint_system.CreateInstanceColumn();
  rotation = Rotation::Cur();
  EXPECT_EQ(constraint_system.QueryInstanceIndex(column, rotation), 1);

  rotation = Rotation::Next();
  EXPECT_EQ(constraint_system.QueryInstanceIndex(column, rotation), 2);
}

TEST_F(ConstraintSystemTest, Phases) {
  ConstraintSystem<math::GF7> constraint_system;
  EXPECT_DEATH(constraint_system.CreateAdviceColumn(kSecondPhase), "");

  std::vector<Phase> phases = {kFirstPhase};
  EXPECT_EQ(constraint_system.ComputeMaxPhase(), kFirstPhase);
  EXPECT_EQ(constraint_system.GetPhases(), phases);

  constraint_system.CreateAdviceColumn(kFirstPhase);
  EXPECT_EQ(constraint_system.ComputeMaxPhase(), kFirstPhase);
  EXPECT_EQ(constraint_system.GetPhases(), phases);

  constraint_system.CreateAdviceColumn(kSecondPhase);
  phases.push_back(kSecondPhase);
  EXPECT_EQ(constraint_system.ComputeMaxPhase(), kSecondPhase);
  EXPECT_EQ(constraint_system.GetPhases(), phases);
}

namespace {

template <typename ColumnKey>
class ConstraintSystemTypedTest : public math::FiniteFieldTest<math::GF7> {};

}  // namespace

using ColumnKeyTypes =
    testing::Types<FixedColumnKey, AdviceColumnKey, InstanceColumnKey>;
TYPED_TEST_SUITE(ConstraintSystemTypedTest, ColumnKeyTypes);

TYPED_TEST(ConstraintSystemTypedTest, EnableEquality) {
  using ColumnKey = TypeParam;

  ConstraintSystem<math::GF7> constraint_system;
  std::vector<AnyColumnKey> expected_permutation_columns;
  std::vector<FixedQueryData> fixed_queries;
  std::vector<AdviceQueryData> advice_queries;
  std::vector<RowIndex> num_advice_queries;
  std::vector<InstanceQueryData> instance_queries;
  EXPECT_EQ(constraint_system.permutation().columns(),
            expected_permutation_columns);

  ColumnKey column;
  if constexpr (std::is_same_v<ColumnKey, FixedColumnKey>) {
    column = constraint_system.CreateFixedColumn();
    fixed_queries.push_back(FixedQueryData(Rotation::Cur(), column));
  } else if constexpr (std::is_same_v<ColumnKey, AdviceColumnKey>) {
    column = constraint_system.CreateAdviceColumn();
    num_advice_queries.push_back(0);
    EXPECT_EQ(constraint_system.num_advice_queries(), num_advice_queries);
    ++num_advice_queries[column.index()];
    advice_queries.push_back(AdviceQueryData(Rotation::Cur(), column));
  } else {
    column = constraint_system.CreateInstanceColumn();
    instance_queries.push_back(InstanceQueryData(Rotation::Cur(), column));
  }
  constraint_system.EnableEquality(column);
  expected_permutation_columns.push_back(AnyColumnKey(column));
  EXPECT_EQ(constraint_system.permutation().columns(),
            expected_permutation_columns);
  EXPECT_EQ(constraint_system.fixed_queries(), fixed_queries);
  EXPECT_EQ(constraint_system.advice_queries(), advice_queries);
  EXPECT_EQ(constraint_system.num_advice_queries(), num_advice_queries);
  EXPECT_EQ(constraint_system.instance_queries(), instance_queries);

  constraint_system.EnableEquality(column);
  EXPECT_EQ(constraint_system.permutation().columns(),
            expected_permutation_columns);
  EXPECT_EQ(constraint_system.fixed_queries(), fixed_queries);
  EXPECT_EQ(constraint_system.advice_queries(), advice_queries);
  EXPECT_EQ(constraint_system.num_advice_queries(), num_advice_queries);
  EXPECT_EQ(constraint_system.instance_queries(), instance_queries);
}

TYPED_TEST(ConstraintSystemTypedTest, QueryAnyIndex) {
  using ColumnKey = TypeParam;

  ConstraintSystem<math::GF7> constraint_system;
  std::function<ColumnKey()> create_column = [&constraint_system]() {
    if constexpr (std::is_same_v<ColumnKey, FixedColumnKey>) {
      return constraint_system.CreateFixedColumn();
    } else if constexpr (std::is_same_v<ColumnKey, AdviceColumnKey>) {
      return constraint_system.CreateAdviceColumn();
    } else {
      return constraint_system.CreateInstanceColumn();
    }
  };

  ColumnKey column = create_column();
  Rotation rotation = Rotation::Cur();
  EXPECT_EQ(constraint_system.QueryAnyIndex(column, rotation), 0);
  EXPECT_EQ(constraint_system.QueryAnyIndex(column, rotation), 0);

  column = create_column();
  rotation = Rotation::Cur();
  EXPECT_EQ(constraint_system.QueryAnyIndex(column, rotation), 1);

  rotation = Rotation::Next();
  EXPECT_EQ(constraint_system.QueryAnyIndex(column, rotation), 2);
}

}  // namespace tachyon::zk::plonk
