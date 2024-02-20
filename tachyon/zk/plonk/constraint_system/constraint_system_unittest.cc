#include "tachyon/zk/plonk/constraint_system/constraint_system.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk::plonk {

namespace {

using F = math::GF7;

class ConstraintSystemTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(ConstraintSystemTest, EnableConstant) {
  ConstraintSystem<F> constraint_system;
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

TEST_F(ConstraintSystemTest, Lookup) {
  ConstraintSystem<F> constraint_system;
  std::array<AdviceColumnKey, 2> advice = {
      constraint_system.CreateAdviceColumn(),
      constraint_system.CreateAdviceColumn(),
  };
  std::array<LookupTableColumn, 2> table = {
      constraint_system.CreateLookupTableColumn(),
      constraint_system.CreateLookupTableColumn(),
  };

  Selector simple_selector = constraint_system.CreateSimpleSelector();
  EXPECT_DEATH(
      constraint_system.Lookup(
          "lookup",
          [simple_selector, &advice, &table](VirtualCells<F>& cells) {
            std::unique_ptr<Expression<F>> simple_selector_expr =
                cells.QuerySelector(simple_selector);
            std::unique_ptr<Expression<F>> advice0_expr =
                cells.QueryAdvice(advice[0], Rotation::Cur());

            LookupPairs<std::unique_ptr<Expression<F>>, LookupTableColumn>
                lookup_pairs;
            lookup_pairs.emplace_back(
                std::move(simple_selector_expr) * std::move(advice0_expr),
                table[0]);
            return lookup_pairs;
          }),
      "expression containing simple selector supplied to lookup argument");

  Selector complex_selector = constraint_system.CreateComplexSelector();
  EXPECT_EQ(constraint_system.Lookup(
                "lookup",
                [complex_selector, &advice, &table](VirtualCells<F>& cells) {
                  std::unique_ptr<Expression<F>> complex_selector_expr =
                      cells.QuerySelector(complex_selector);
                  std::unique_ptr<Expression<F>> advice0_expr =
                      cells.QueryAdvice(advice[0], Rotation::Cur());

                  LookupPairs<std::unique_ptr<Expression<F>>, LookupTableColumn>
                      lookup_pairs;
                  lookup_pairs.emplace_back(std::move(complex_selector_expr) *
                                                std::move(advice0_expr),
                                            table[0]);
                  return lookup_pairs;
                }),
            0);

  EXPECT_EQ(
      constraint_system.Lookup(
          "lookup",
          [complex_selector, &advice, &table](VirtualCells<F>& cells) {
            std::unique_ptr<Expression<F>> complex_selector_expr =
                cells.QuerySelector(complex_selector);
            std::unique_ptr<Expression<F>> advice1_expr =
                cells.QueryAdvice(advice[1], Rotation::Cur());
            std::unique_ptr<Expression<F>> not_complex_selector_expr =
                ExpressionFactory<F>::Constant(F::One()) -
                complex_selector_expr->Clone();
            std::unique_ptr<Expression<F>> default_expr =
                ExpressionFactory<F>::Constant(F(2));

            LookupPairs<std::unique_ptr<Expression<F>>, LookupTableColumn>
                lookup_pairs;
            lookup_pairs.emplace_back(
                std::move(complex_selector_expr) * std::move(advice1_expr) +
                    std::move(not_complex_selector_expr) *
                        std::move(default_expr),
                table[1]);
            return lookup_pairs;
          }),
      1);

  EXPECT_EQ(constraint_system.ComputeLookupRequiredDegree(), 5);

  std::vector<LookupArgument<F>> expected_lookups;
  {
    LookupPairs<std::unique_ptr<Expression<F>>> pairs;
    pairs.emplace_back(ExpressionFactory<F>::Product(
                           ExpressionFactory<F>::Selector(complex_selector),
                           ExpressionFactory<F>::Advice(
                               AdviceQuery(0, Rotation::Cur(), advice[0]))),
                       ExpressionFactory<F>::Fixed(
                           FixedQuery(0, Rotation::Cur(), table[0].column())));
    expected_lookups.emplace_back("lookup", std::move(pairs));

    LookupPairs<std::unique_ptr<Expression<F>>> pairs2;
    pairs2.emplace_back(
        ExpressionFactory<F>::Sum(
            ExpressionFactory<F>::Product(
                ExpressionFactory<F>::Selector(complex_selector),
                ExpressionFactory<F>::Advice(
                    AdviceQuery(1, Rotation::Cur(), advice[1]))),
            ExpressionFactory<F>::Product(
                ExpressionFactory<F>::Sum(
                    ExpressionFactory<F>::Constant(F::One()),
                    ExpressionFactory<F>::Negated(
                        ExpressionFactory<F>::Selector(complex_selector))),
                ExpressionFactory<F>::Constant(F(2)))),
        ExpressionFactory<F>::Fixed(
            FixedQuery(1, Rotation::Cur(), table[1].column())));
    expected_lookups.emplace_back("lookup", std::move(pairs2));
  }
  EXPECT_EQ(constraint_system.lookups(), expected_lookups);
}

TEST_F(ConstraintSystemTest, LookupAny) {
  ConstraintSystem<F> constraint_system;
  AdviceColumnKey advice = constraint_system.CreateAdviceColumn();
  InstanceColumnKey table = constraint_system.CreateInstanceColumn();
  AdviceColumnKey advice_table = constraint_system.CreateAdviceColumn();

  Selector simple_selector = constraint_system.CreateSimpleSelector();
  EXPECT_DEATH(
      constraint_system.LookupAny(
          "lookup",
          [&advice, simple_selector, &advice_table,
           &table](VirtualCells<F>& cells) {
            std::unique_ptr<Expression<F>> advice_expr =
                cells.QueryAdvice(advice, Rotation::Cur());
            std::unique_ptr<Expression<F>> simple_selector_expr =
                cells.QuerySelector(simple_selector);
            std::unique_ptr<Expression<F>> advice_table_expr =
                cells.QueryAdvice(advice_table, Rotation::Cur());
            std::unique_ptr<Expression<F>> table_expr =
                cells.QueryInstance(table, Rotation::Cur());

            LookupPairs<std::unique_ptr<Expression<F>>> lookup_pairs;
            lookup_pairs.emplace_back(
                simple_selector_expr->Clone() * advice_expr->Clone(),
                std::move(table_expr));
            lookup_pairs.emplace_back(
                std::move(simple_selector_expr) * std::move(advice_expr),
                std::move(advice_table_expr));
            return lookup_pairs;
          }),
      "expression containing simple selector supplied to lookup argument");

  Selector complex_selector = constraint_system.CreateComplexSelector();
  EXPECT_EQ(
      constraint_system.LookupAny(
          "lookup",
          [&advice, complex_selector, &advice_table,
           &table](VirtualCells<F>& cells) {
            std::unique_ptr<Expression<F>> advice_expr =
                cells.QueryAdvice(advice, Rotation::Cur());
            std::unique_ptr<Expression<F>> complex_selector_expr =
                cells.QuerySelector(complex_selector);
            std::unique_ptr<Expression<F>> not_complex_selector_expr =
                ExpressionFactory<F>::Constant(F::One()) -
                complex_selector_expr->Clone();
            std::unique_ptr<Expression<F>> default_expr =
                ExpressionFactory<F>::Constant(F(2));
            std::unique_ptr<Expression<F>> advice_table_expr =
                cells.QueryAdvice(advice_table, Rotation::Cur());
            std::unique_ptr<Expression<F>> table_expr =
                cells.QueryInstance(table, Rotation::Cur());

            LookupPairs<std::unique_ptr<Expression<F>>> lookup_pairs;
            lookup_pairs.emplace_back(
                complex_selector_expr->Clone() * advice_expr->Clone() +
                    not_complex_selector_expr->Clone() * default_expr->Clone(),
                std::move(table_expr));
            lookup_pairs.emplace_back(
                std::move(complex_selector_expr) * std::move(advice_expr) +
                    std::move(not_complex_selector_expr) *
                        std::move(default_expr),
                std::move(advice_table_expr));
            return lookup_pairs;
          }),
      0);

  EXPECT_EQ(constraint_system.ComputeLookupRequiredDegree(), 5);

  std::vector<LookupArgument<F>> expected_lookups;
  {
    LookupPairs<std::unique_ptr<Expression<F>>> pairs;
    pairs.emplace_back(
        ExpressionFactory<F>::Sum(
            ExpressionFactory<F>::Product(
                ExpressionFactory<F>::Selector(complex_selector),
                ExpressionFactory<F>::Advice(
                    AdviceQuery(0, Rotation::Cur(), advice))),
            ExpressionFactory<F>::Product(
                ExpressionFactory<F>::Sum(
                    ExpressionFactory<F>::Constant(F::One()),
                    ExpressionFactory<F>::Negated(
                        ExpressionFactory<F>::Selector(complex_selector))),
                ExpressionFactory<F>::Constant(F(2)))),
        ExpressionFactory<F>::Instance(
            InstanceQuery(0, Rotation::Cur(), table)));
    pairs.emplace_back(
        ExpressionFactory<F>::Sum(
            ExpressionFactory<F>::Product(
                ExpressionFactory<F>::Selector(complex_selector),
                ExpressionFactory<F>::Advice(
                    AdviceQuery(0, Rotation::Cur(), advice))),
            ExpressionFactory<F>::Product(
                ExpressionFactory<F>::Sum(
                    ExpressionFactory<F>::Constant(F::One()),
                    ExpressionFactory<F>::Negated(
                        ExpressionFactory<F>::Selector(complex_selector))),
                ExpressionFactory<F>::Constant(F(2)))),
        ExpressionFactory<F>::Advice(
            AdviceQuery(1, Rotation::Cur(), advice_table)));
    expected_lookups.emplace_back("lookup", std::move(pairs));
  }
  EXPECT_EQ(constraint_system.lookups(), expected_lookups);
}

TEST_F(ConstraintSystemTest, QueryFixedIndex) {
  ConstraintSystem<F> constraint_system;
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
  ConstraintSystem<F> constraint_system;
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
  ConstraintSystem<F> constraint_system;
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
  ConstraintSystem<F> constraint_system;
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
class ConstraintSystemTypedTest : public math::FiniteFieldTest<F> {};

}  // namespace

using ColumnKeyTypes =
    testing::Types<FixedColumnKey, AdviceColumnKey, InstanceColumnKey>;
TYPED_TEST_SUITE(ConstraintSystemTypedTest, ColumnKeyTypes);

TYPED_TEST(ConstraintSystemTypedTest, EnableEquality) {
  using ColumnKey = TypeParam;

  ConstraintSystem<F> constraint_system;
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

  ConstraintSystem<F> constraint_system;
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
