#include "tachyon/zk/plonk/circuit/expressions/evaluator/simple_selector_finder.h"

#include <memory>

#include "tachyon/zk/plonk/circuit/expressions/evaluator/test/evaluator_test.h"
#include "tachyon/zk/plonk/circuit/expressions/expression_factory.h"

namespace tachyon::zk {

using Expr = std::unique_ptr<Expression<GF7>>;

TEST_F(EvaluatorTest, Constant) {
  GF7 value = GF7::Random();
  std::unique_ptr<Expression<GF7>> expr =
      ExpressionFactory<GF7>::Constant(value);
  EXPECT_FALSE(expr->ContainsSimpleSelector());
}

TEST_F(EvaluatorTest, Selector) {
  Expr expr = ExpressionFactory<GF7>::Selector(Selector::Simple(1));
  EXPECT_TRUE(expr->ContainsSimpleSelector());
  expr = ExpressionFactory<GF7>::Selector(Selector::Complex(1));
  EXPECT_FALSE(expr->ContainsSimpleSelector());
}

TEST_F(EvaluatorTest, Fixed) {
  struct {
    size_t column_index;
    int32_t rotation;
  } tests[] = {
      {0, 1},
      {1, 2},
  };

  for (const auto& test : tests) {
    FixedQuery query(1, Rotation(test.rotation),
                     FixedColumn(test.column_index));
    Expr expr = ExpressionFactory<GF7>::Fixed(query);
    EXPECT_FALSE(expr->ContainsSimpleSelector());
  }
}

TEST_F(EvaluatorTest, Advice) {
  struct {
    size_t column_index;
    int32_t rotation;
  } tests[] = {
      {2, 6},
      {3, 7},
  };

  for (const auto& test : tests) {
    AdviceQuery query(1, Rotation(test.rotation),
                      AdviceColumn(test.column_index, Phase(0)));
    Expr expr = ExpressionFactory<GF7>::Advice(query);
    EXPECT_FALSE(expr->ContainsSimpleSelector());
  }
}

TEST_F(EvaluatorTest, Instance) {
  struct {
    size_t column_index;
    int32_t rotation;
  } tests[] = {
      {1, 1},
      {2, 2},
  };

  for (const auto& test : tests) {
    InstanceQuery query(1, Rotation(test.rotation),
                        InstanceColumn(test.column_index));
    Expr expr = ExpressionFactory<GF7>::Instance(query);
    EXPECT_FALSE(expr->ContainsSimpleSelector());
  }
}

TEST_F(EvaluatorTest, Challenges) {
  for (size_t i = 0; i < challenges_.size(); ++i) {
    Expr expr = ExpressionFactory<GF7>::Challenge(Challenge(i, Phase(0)));
    EXPECT_FALSE(expr->ContainsSimpleSelector());
  }
}

TEST_F(EvaluatorTest, Negated) {
  GF7 value = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Negated(ExpressionFactory<GF7>::Constant(value));
  EXPECT_FALSE(expr->ContainsSimpleSelector());
}

TEST_F(EvaluatorTest, Sum) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Sum({ExpressionFactory<GF7>::Constant(a),
                                   ExpressionFactory<GF7>::Constant(b)});
  EXPECT_FALSE(expr->ContainsSimpleSelector());
  expr = ExpressionFactory<GF7>::Sum(
      {ExpressionFactory<GF7>::Constant(a),
       ExpressionFactory<GF7>::Selector(Selector::Simple(1))});
  EXPECT_TRUE(expr->ContainsSimpleSelector());
}

TEST_F(EvaluatorTest, Product) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Product({ExpressionFactory<GF7>::Constant(a),
                                       ExpressionFactory<GF7>::Constant(b)});
  EXPECT_FALSE(expr->ContainsSimpleSelector());
  expr = ExpressionFactory<GF7>::Product(
      {ExpressionFactory<GF7>::Constant(a),
       ExpressionFactory<GF7>::Selector(Selector::Simple(1))});
  EXPECT_TRUE(expr->ContainsSimpleSelector());
}

TEST_F(EvaluatorTest, Scaled) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Scaled({ExpressionFactory<GF7>::Constant(a), b});
  EXPECT_FALSE(expr->ContainsSimpleSelector());
  expr = ExpressionFactory<GF7>::Scaled(
      {ExpressionFactory<GF7>::Selector(Selector::Simple(1)), GF7(3)});
  EXPECT_TRUE(expr->ContainsSimpleSelector());
}

}  // namespace tachyon::zk
