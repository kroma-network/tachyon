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

TEST_F(GraphEvaluatorTest, ConstantNegated) {
  GraphEvaluator<GF7> graph_evaluator;
  for (size_t i = 0; i < 7; ++i) {
    std::unique_ptr<Expression<GF7>> expr = ExpressionFactory<GF7>::Negated(
        ExpressionFactory<GF7>::Constant(GF7(i)));
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    if (i == 0) {
      EXPECT_TRUE(source.IsZeroConstant());
    } else if (i == 6) {
      // Negation of 6 in GF7 is 1.
      EXPECT_EQ(source, ValueSource::OneConstant());
    } else if (i == 5) {
      // Negation of 5 in GF7 is 2.
      EXPECT_EQ(source, ValueSource::TwoConstant());
    } else {
      // Other constants besides 0, 1, and 2 are stored sequentially regardless
      // of their value.
      EXPECT_EQ(source, ValueSource::Constant(i + 2));
    }
  }
}

TEST_F(GraphEvaluatorTest, Negated) {
  std::vector<ColumnInfo> test_column_infos = GenerateTestColumnInfos();
  // clang-format off
  // 1. test_column_infos[0]
  //      = {column_index: 0, rotation: 1, rotation_index: 0, calculation_index: 0}
  //    expected |calculations_|
  //      = {test_column_infos[0], Negate(test_column_infos[0])}

  // 2. test_column_infos[1]
  //      = {column_index: 1, rotation: 2, rotation_index: 1, calculation_index: 1}
  //    expected |calculations_|
  //      = {test_column_infos[0], Negate(test_column_infos[0]), test_column_infos[1], Negate(test_column_infos[1])}

  // 3. test_column_infos[2] (test_column_infos[2] is same as test_column_infos[0])
  //      = {column_index: 0, rotation: 1, rotation_index: 0, calculation_index: 0}
  //    expected |calculations_|
  //      = {test_column_infos[0], Negate(test_column_infos[0]), test_column_infos[1], Negate(test_column_infos[1])}

  // 4. test_column_infos[3]
  //      = {column_index: 0, rotation: 2, rotation_index: 1, calculation_index: 2}
  //    expected |calculations_|
  //      = {test_column_infos[0], Negate(test_column_infos[0]), test_column_infos[1], Negate(test_column_infos[1]), test_column_infos[3], Negate(test_column_infos[3])}
  // clang-format on
  std::vector<size_t> expected_indices = {1, 3, 1, 5};

  GraphEvaluator<GF7> graph_evaluator;
  for (size_t i = 0; i < test_column_infos.size(); ++i) {
    FixedQuery query(1, Rotation(test_column_infos[i].rotation),
                     FixedColumnKey(test_column_infos[i].column_index));
    std::unique_ptr<Expression<GF7>> expr =
        ExpressionFactory<GF7>::Negated(ExpressionFactory<GF7>::Fixed(query));
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    EXPECT_EQ(source, ValueSource::Intermediate(expected_indices[i]));

    EXPECT_EQ(graph_evaluator.calculations()[expected_indices[i]],
              CalculationInfo(Calculation::Negate(ValueSource::Intermediate(
                                  expected_indices[i] - 1)),
                              expected_indices[i]));
    EXPECT_EQ(graph_evaluator.calculations()[expected_indices[i] - 1],
              CalculationInfo(Calculation::Store(ValueSource::Fixed(
                                  test_column_infos[i].column_index,
                                  test_column_infos[i].rotation_index)),
                              expected_indices[i] - 1));
  }

  EXPECT_EQ(graph_evaluator.rotations().size(), 2);
  EXPECT_EQ(graph_evaluator.rotations()[0], test_column_infos[0].rotation);
  EXPECT_EQ(graph_evaluator.rotations()[1], test_column_infos[1].rotation);
}

TEST_F(GraphEvaluatorTest, NegatedSum) {
  struct {
    int left;
    int right;
    int expected_left_index;
    int expected_right_index;
  } tests[]{
      // default |constants_| = {0, 1, 2}

      // Case 1(left = 0):
      {0, 4, 0, 3},
      // expected |constants_| = {0, 1, 2, 4}
      // expected |calculations_| = {Negate(4)}

      // Case 2(right = 0):
      {3, 0, 4, 0},
      // expected |constants_| = {0, 1, 2, 4, 3}
      // expected |calculations_| = {Negate(4)}

      // Case 3:
      {2, 6, 2, 5},
      // expected |constants_| = {0, 1, 2, 4, 3, 6}
      // expected |calculations_| = {Negate(4), Sub(2, 6)}
      {5, 4, 6, 3},
      // expected |constants_| = {0, 1, 2, 4, 3, 6, 5}
      // expected |calculations_| = {Negate(4), Sub(2, 6), Sub(5, 4)}
  };

  size_t expected_calculations = 0;

  GraphEvaluator<GF7> graph_evaluator;
  for (const auto& test : tests) {
    std::unique_ptr<Expression<GF7>> expr = ExpressionFactory<GF7>::Sum(
        ExpressionFactory<GF7>::Constant(GF7(test.left)),
        ExpressionFactory<GF7>::Negated(
            ExpressionFactory<GF7>::Constant(GF7(test.right))));
    ValueSource source = graph_evaluator.Evaluate(expr.get());

    if (test.left == 0) {
      // Case 1(left = 0):
      EXPECT_EQ(source, ValueSource::Intermediate(expected_calculations));
      EXPECT_EQ(graph_evaluator.calculations()[expected_calculations],
                CalculationInfo(Calculation::Negate(ValueSource::Constant(
                                    test.expected_right_index)),
                                expected_calculations));
      ++expected_calculations;
    } else if (test.right == 0) {
      // Case 2(right = 0):
      EXPECT_EQ(source, ValueSource::Constant(test.expected_left_index));
    } else {
      // Case 3:
      EXPECT_EQ(source, ValueSource::Intermediate(expected_calculations));
      EXPECT_EQ(
          graph_evaluator.calculations()[expected_calculations],
          CalculationInfo(Calculation::Sub(
                              ValueSource::Constant(test.expected_left_index),
                              ValueSource::Constant(test.expected_right_index)),
                          expected_calculations));
      ++expected_calculations;
    }
    EXPECT_EQ(graph_evaluator.num_intermediates(), expected_calculations);
  }
}

TEST_F(GraphEvaluatorTest, ConstantSum) {
  struct {
    int left;
    int right;
    int expected_left_index;
    int expected_right_index;
  } tests[]{
      // default |constants_| = {0, 1, 2}

      // Case 1(left = 0):
      {0, 4, 0, 3},
      // expected |constants_| = {0, 1, 2, 4}
      // expected |calculations_| = {}

      // Case 2(right = 0):
      {3, 0, 4, 0},
      // expected |constants_| = {0, 1, 2, 4, 3}
      // expected |calculations_| = {}

      // Case 3(left <= right):
      {2, 6, 2, 5},
      // expected |constants_| = {0, 1, 2, 4, 3, 6}
      // expected |calculations_| = {Add(2, 6)}

      // Case 4(left > right):
      {5, 4, 6, 3},
      // expected |constants_| = {0, 1, 2, 4, 3, 6, 5}
      // expected |calculations_| = {Add(2, 6), Add(5, 4)}
  };

  size_t expected_calculations = 0;

  GraphEvaluator<GF7> graph_evaluator;
  for (const auto& test : tests) {
    std::unique_ptr<Expression<GF7>> expr = ExpressionFactory<GF7>::Sum(
        ExpressionFactory<GF7>::Constant(GF7(test.left)),
        ExpressionFactory<GF7>::Constant(GF7(test.right)));
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    if (test.left == 0) {
      // Case 1(left = 0):
      EXPECT_EQ(source, ValueSource::Constant(test.expected_right_index));
    } else if (test.right == 0) {
      // Case 2(right = 0):
      EXPECT_EQ(source, ValueSource::Constant(test.expected_left_index));
    } else {
      EXPECT_EQ(source, ValueSource::Intermediate(expected_calculations));
      if (test.left <= test.right) {
        // Case 3(left <= right):
        EXPECT_EQ(graph_evaluator.calculations()[expected_calculations],
                  CalculationInfo(
                      Calculation::Add(
                          ValueSource::Constant(test.expected_left_index),
                          ValueSource::Constant(test.expected_right_index)),
                      expected_calculations));
      } else {
        // Case 4(left > right):
        EXPECT_EQ(graph_evaluator.calculations()[expected_calculations],
                  CalculationInfo(
                      Calculation::Add(
                          ValueSource::Constant(test.expected_right_index),
                          ValueSource::Constant(test.expected_left_index)),
                      expected_calculations));
      }
      ++expected_calculations;
    }
    EXPECT_EQ(graph_evaluator.num_intermediates(), expected_calculations);
  }
}

TEST_F(GraphEvaluatorTest, ConstantProduct) {
  struct {
    int left;
    int right;
    int expected_left_index;
    int expected_right_index;
  } tests[]{
      // default |constants_| = {0, 1, 2}

      // Case 1(left = 0 || right = 0):
      {0, 3, 0, 3},
      // expected |constants_| = {0, 1, 2, 3}
      // expected |calculations_| = {}
      {5, 0, 4, 0},
      // expected |constants_| = {0, 1, 2, 3, 5}
      // expected |calculations_| = {}

      // Case 2(left = 1):
      {1, 2, 1, 2},
      // expected |constants_| = {0, 1, 2, 3, 5}
      // expected |calculations_| = {}

      // Case 3(right = 1):
      {3, 1, 3, 1},
      // expected |constants_| = {0, 1, 2, 3, 5}
      // expected |calculations_| = {}

      // Case 4(left = 2):
      {2, 4, 2, 5},
      // expected |constants_| = {0, 1, 2, 3, 5, 4}
      // expected |calculations_| = {Double(4)}

      // Case 5(right = 2):
      {3, 2, 3, 2},
      // expected |constants_| = {0, 1, 2, 3, 5, 4}
      // expected |calculations_| = {Double(4), Double(3)}

      // Case 6(left = right):
      {5, 5, 4, 4},
      // expected |constants_| = {0, 1, 2, 3, 5, 4}
      // expected |calculations_| = {Double(4), Double(3), Square(5)}

      // Case 7:
      {4, 6, 5, 6},
      // expected |constants_| = {0, 1, 2, 3, 5, 4, 6}
      // expected |calculations_| = {Double(4), Double(3), Square(5), Mul(4, 6)}
      // expected size of |calculations_| = 4
  };

  size_t expected_calculations = 0;

  GraphEvaluator<GF7> graph_evaluator;
  for (const auto& test : tests) {
    std::unique_ptr<Expression<GF7>> expr = ExpressionFactory<GF7>::Product(
        ExpressionFactory<GF7>::Constant(GF7(test.left)),
        ExpressionFactory<GF7>::Constant(GF7(test.right)));
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    if (test.left == 0 || test.right == 0) {
      // Case 1(left = 0 || right = 0):
      EXPECT_EQ(source, ValueSource::ZeroConstant());
    } else if (test.left == 1) {
      // Case 2(left = 1):
      EXPECT_EQ(source, ValueSource::Constant(test.expected_right_index));
    } else if (test.right == 1) {
      // Case 3(right = 1):
      EXPECT_EQ(source, ValueSource::Constant(test.expected_left_index));
    } else {
      EXPECT_EQ(source, ValueSource::Intermediate(expected_calculations));
      if (test.left == 2) {
        // Case 4(left = 2):
        EXPECT_EQ(graph_evaluator.calculations()[expected_calculations],
                  CalculationInfo(Calculation::Double(ValueSource::Constant(
                                      test.expected_right_index)),
                                  expected_calculations));
      } else if (test.right == 2) {
        // Case 5(right = 2):
        EXPECT_EQ(graph_evaluator.calculations()[expected_calculations],
                  CalculationInfo(Calculation::Double(ValueSource::Constant(
                                      test.expected_left_index)),
                                  expected_calculations));
      } else if (test.left == test.right) {
        // Case 6(left = right):
        EXPECT_EQ(graph_evaluator.calculations()[expected_calculations],
                  CalculationInfo(Calculation::Square(ValueSource::Constant(
                                      test.expected_left_index)),
                                  expected_calculations));
      } else {
        // Case 7:
        EXPECT_EQ(graph_evaluator.calculations()[expected_calculations],
                  CalculationInfo(
                      Calculation::Mul(
                          ValueSource::Constant(test.expected_left_index),
                          ValueSource::Constant(test.expected_right_index)),
                      expected_calculations));
      }
      ++expected_calculations;
      EXPECT_EQ(graph_evaluator.num_intermediates(), expected_calculations);
    }
  }
}

TEST_F(GraphEvaluatorTest, Scaled) {
  struct {
    int expr;
    int scalar;
    int expected_expr_index;
    int expected_scalar_index;
  } tests[] = {
      // default |constants_| = {0, 1, 2}

      // Case 1(scalar = 0):
      {2, 0, 2, 0},
      // expected |constants_| = {0, 1, 2}
      // expected |calculations_| = {}

      // Case 2(scalar = 1):
      {1, 1, 1, 1},
      // expected |constants_| = {0, 1, 2}
      // expected |calculations_| = {}
      {4, 1, 3, 1},
      // expected |constants_| = {0, 1, 2, 4}
      // expected |calculations_| = {}

      // Case 3:
      {2, 2, 2, 2},
      // expected |constants_| = {0, 1, 2, 4}
      // expected |calculations_| = {Mul(2, 2)}
      {6, 3, 5, 4},
      // expected |constants_| = {0, 1, 2, 4, 6}
      // expected |calculations_| = {Mul(2, 2), Mul(6, 3)}
  };

  size_t expected_calculations = 0;
  GraphEvaluator<GF7> graph_evaluator;
  for (const auto& test : tests) {
    std::unique_ptr<Expression<GF7>> expr = ExpressionFactory<GF7>::Scaled(
        ExpressionFactory<GF7>::Constant(GF7(test.expr)), GF7(test.scalar));
    ValueSource source = graph_evaluator.Evaluate(expr.get());
    if (test.scalar == 0) {
      // Case 1(scalar = 0):
      EXPECT_EQ(source, ValueSource::ZeroConstant());
    } else if (test.scalar == 1) {
      // Case 2(scalar = 1):
      EXPECT_EQ(source, ValueSource::Constant(test.expected_expr_index));
    } else {
      // Case 3:
      EXPECT_EQ(source, ValueSource::Intermediate(expected_calculations));
      EXPECT_EQ(graph_evaluator.calculations()[expected_calculations],
                CalculationInfo(
                    Calculation::Mul(
                        ValueSource::Constant(test.expected_expr_index),
                        ValueSource::Constant(test.expected_scalar_index)),
                    expected_calculations));
      ++expected_calculations;
    }
    EXPECT_EQ(graph_evaluator.num_intermediates(), expected_calculations);
  }
}

}  // namespace tachyon::zk
