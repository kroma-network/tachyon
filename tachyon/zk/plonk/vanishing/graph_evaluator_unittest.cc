#include "tachyon/zk/plonk/vanishing/graph_evaluator.h"

#include <memory>

#include "tachyon/base/random.h"
#include "tachyon/zk/expressions/evaluator/test/evaluator_test.h"
#include "tachyon/zk/expressions/expression_factory.h"

namespace tachyon::zk {

namespace {

using Expr = std::unique_ptr<Expression<GF7>>;

struct ColumnInfo {
  size_t column_index;
  int32_t rotation;
  size_t rotation_index;
  size_t calculation_index;
};

std::vector<ColumnInfo> GenerateTestColumnInfos() {
  std::vector<ColumnInfo> test_column_infos;
  test_column_infos.push_back({0, 1, 0, 0});
  test_column_infos.push_back({1, 2, 1, 1});
  test_column_infos.push_back({0, 1, 0, 0});
  test_column_infos.push_back({0, 2, 1, 2});
  return test_column_infos;
}

class GraphEvaluatorTest : public EvaluatorTest {};

}  // namespace

TEST_F(GraphEvaluatorTest, Constant) {
  GraphEvaluator<GF7> graph_evaluator;
  int values[] = {0, 1, 2, base::Uniform(base::Range<int>(3, 7))};
  for (int value : values) {
    std::unique_ptr<Expression<GF7>> expr =
        ExpressionFactory<GF7>::Constant(GF7(value));
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    if (value == 0) {
      EXPECT_TRUE(source.IsZeroConstant());
    } else if (value == 1) {
      EXPECT_TRUE(source.IsOneConstant());
    } else if (value == 2) {
      EXPECT_TRUE(source.IsTwoConstant());
    } else {
      EXPECT_EQ(source, ValueSource::Constant(3));
    }
  }
}

TEST_F(GraphEvaluatorTest, Selector) {
  Expr expr = ExpressionFactory<GF7>::Selector(Selector::Simple(1));
  GraphEvaluator<GF7> graph_evaluator;
  EXPECT_DEATH(graph_evaluator.Evaluate(expr.get()), "");
}

TEST_F(GraphEvaluatorTest, Fixed) {
  std::vector<ColumnInfo> test_column_infos = GenerateTestColumnInfos();

  GraphEvaluator<GF7> graph_evaluator;
  for (const ColumnInfo& column_info : test_column_infos) {
    FixedQuery query(1, Rotation(column_info.rotation),
                     FixedColumnKey(column_info.column_index));
    Expr expr = ExpressionFactory<GF7>::Fixed(query);
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    EXPECT_EQ(source, ValueSource::Intermediate(column_info.calculation_index));
    EXPECT_EQ(graph_evaluator.calculations()[column_info.calculation_index],
              CalculationInfo(
                  Calculation::Store(ValueSource::Fixed(
                      column_info.column_index, column_info.rotation_index)),
                  column_info.calculation_index));
    EXPECT_EQ(graph_evaluator.rotations()[column_info.rotation_index],
              column_info.rotation);
  }
}

TEST_F(GraphEvaluatorTest, Advice) {
  std::vector<ColumnInfo> test_column_infos = GenerateTestColumnInfos();

  GraphEvaluator<GF7> graph_evaluator;
  for (const ColumnInfo& column_info : test_column_infos) {
    AdviceQuery query(1, Rotation(column_info.rotation),
                      AdviceColumnKey(column_info.column_index));
    Expr expr = ExpressionFactory<GF7>::Advice(query);
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    EXPECT_EQ(source, ValueSource::Intermediate(column_info.calculation_index));
    EXPECT_EQ(graph_evaluator.calculations()[column_info.calculation_index],
              CalculationInfo(
                  Calculation::Store(ValueSource::Advice(
                      column_info.column_index, column_info.rotation_index)),
                  column_info.calculation_index));
    EXPECT_EQ(graph_evaluator.rotations()[column_info.rotation_index],
              column_info.rotation);
  }
}

TEST_F(GraphEvaluatorTest, Instance) {
  std::vector<ColumnInfo> test_column_infos = GenerateTestColumnInfos();

  GraphEvaluator<GF7> graph_evaluator;
  for (const ColumnInfo& column_info : test_column_infos) {
    InstanceQuery query(1, Rotation(column_info.rotation),
                        InstanceColumnKey(column_info.column_index));
    Expr expr = ExpressionFactory<GF7>::Instance(query);
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    EXPECT_EQ(source, ValueSource::Intermediate(column_info.calculation_index));
    EXPECT_EQ(graph_evaluator.calculations()[column_info.calculation_index],
              CalculationInfo(
                  Calculation::Store(ValueSource::Instance(
                      column_info.column_index, column_info.rotation_index)),
                  column_info.calculation_index));
    EXPECT_EQ(graph_evaluator.rotations()[column_info.rotation_index],
              column_info.rotation);
  }
}

TEST_F(GraphEvaluatorTest, Challenges) {
  GraphEvaluator<GF7> graph_evaluator;
  Expr expr = ExpressionFactory<GF7>::Challenge(Challenge(1, Phase(0)));
  ValueSource source = graph_evaluator.Evaluate(expr.get());
  EXPECT_EQ(source, ValueSource::Intermediate(0));
  EXPECT_EQ(graph_evaluator.calculations()[0],
            CalculationInfo(Calculation::Store(ValueSource::Challenge(1)), 0));
}

// TODO(chokobole): AddTest for Negated, Sum, Product and Scale.

}  // namespace tachyon::zk
