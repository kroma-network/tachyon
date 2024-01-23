#include "tachyon/zk/plonk/vanishing/graph_evaluator.h"

#include <memory>

#include "tachyon/base/random.h"
#include "tachyon/zk/expressions/evaluator/test/evaluator_test.h"
#include "tachyon/zk/expressions/expression_factory.h"

namespace tachyon::zk {

using Expr = std::unique_ptr<Expression<GF7>>;

class GraphEvaluatorTest : public EvaluatorTest {};

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
  struct {
    size_t column_index;
    int32_t rotation;
    size_t rotation_index;
    size_t calculation_index;
  } tests[] = {
      {0, 1, 0, 0},
      {1, 2, 1, 1},
      {0, 1, 0, 0},
      {0, 2, 1, 2},
  };

  GraphEvaluator<GF7> graph_evaluator;
  for (const auto& test : tests) {
    FixedQuery query(1, Rotation(test.rotation),
                     FixedColumnKey(test.column_index));
    Expr expr = ExpressionFactory<GF7>::Fixed(query);
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    EXPECT_EQ(source, ValueSource::Intermediate(test.calculation_index));
    EXPECT_EQ(graph_evaluator.calculations()[test.calculation_index],
              CalculationInfo(Calculation::Store(ValueSource::Fixed(
                                  test.column_index, test.rotation_index)),
                              test.calculation_index));
    EXPECT_EQ(graph_evaluator.rotations()[test.rotation_index], test.rotation);
  }
}

TEST_F(GraphEvaluatorTest, Advice) {
  struct {
    size_t column_index;
    int32_t rotation;
    size_t rotation_index;
    size_t calculation_index;
  } tests[] = {
      {0, 1, 0, 0},
      {1, 2, 1, 1},
      {0, 1, 0, 0},
      {0, 2, 1, 2},
  };

  GraphEvaluator<GF7> graph_evaluator;
  for (const auto& test : tests) {
    AdviceQuery query(1, Rotation(test.rotation),
                      AdviceColumnKey(test.column_index));
    Expr expr = ExpressionFactory<GF7>::Advice(query);
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    EXPECT_EQ(source, ValueSource::Intermediate(test.calculation_index));
    EXPECT_EQ(graph_evaluator.calculations()[test.calculation_index],
              CalculationInfo(Calculation::Store(ValueSource::Advice(
                                  test.column_index, test.rotation_index)),
                              test.calculation_index));
    EXPECT_EQ(graph_evaluator.rotations()[test.rotation_index], test.rotation);
  }
}

TEST_F(GraphEvaluatorTest, Instance) {
  struct {
    size_t column_index;
    int32_t rotation;
    size_t rotation_index;
    size_t calculation_index;
  } tests[] = {
      {0, 1, 0, 0},
      {1, 2, 1, 1},
      {0, 1, 0, 0},
      {0, 2, 1, 2},
  };

  GraphEvaluator<GF7> graph_evaluator;
  for (const auto& test : tests) {
    InstanceQuery query(1, Rotation(test.rotation),
                        InstanceColumnKey(test.column_index));
    Expr expr = ExpressionFactory<GF7>::Instance(query);
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    EXPECT_EQ(source, ValueSource::Intermediate(test.calculation_index));
    EXPECT_EQ(graph_evaluator.calculations()[test.calculation_index],
              CalculationInfo(Calculation::Store(ValueSource::Instance(
                                  test.column_index, test.rotation_index)),
                              test.calculation_index));
    EXPECT_EQ(graph_evaluator.rotations()[test.rotation_index], test.rotation);
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
