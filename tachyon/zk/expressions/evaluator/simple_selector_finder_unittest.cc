#include "tachyon/zk/expressions/evaluator/simple_selector_finder.h"

#include <memory>

#include "tachyon/zk/expressions/evaluator/test/evaluator_test.h"
#include "tachyon/zk/expressions/expression_factory.h"

namespace tachyon::zk {

using Expr = std::unique_ptr<Expression<GF7>>;

class SimpleSelectorFinderTest : public EvaluatorTest {};

TEST_F(SimpleSelectorFinderTest, Constant) {
  GF7 value = GF7::Random();
  std::unique_ptr<Expression<GF7>> expr =
      ExpressionFactory<GF7>::Constant(value);
  EXPECT_FALSE(ContainsSimpleSelector(expr.get()));
}

TEST_F(SimpleSelectorFinderTest, Selector) {
  Expr expr = ExpressionFactory<GF7>::Selector(plonk::Selector::Simple(1));
  EXPECT_TRUE(ContainsSimpleSelector(expr.get()));
  expr = ExpressionFactory<GF7>::Selector(plonk::Selector::Complex(1));
  EXPECT_FALSE(ContainsSimpleSelector(expr.get()));
}

TEST_F(SimpleSelectorFinderTest, Fixed) {
  struct {
    int32_t rotation;
    size_t column_index;
  } tests[] = {
      {1, 0},
      {2, 1},
  };

  for (const auto& test : tests) {
    plonk::FixedQuery query(1, Rotation(test.rotation),
                            plonk::FixedColumnKey(test.column_index));
    Expr expr = ExpressionFactory<GF7>::Fixed(query);
    EXPECT_FALSE(ContainsSimpleSelector(expr.get()));
  }
}

TEST_F(SimpleSelectorFinderTest, Advice) {
  struct {
    int32_t rotation;
    size_t column_index;
  } tests[] = {
      {6, 2},
      {7, 3},
  };

  for (const auto& test : tests) {
    plonk::AdviceQuery query(
        1, Rotation(test.rotation),
        plonk::AdviceColumnKey(test.column_index, plonk::Phase(0)));
    Expr expr = ExpressionFactory<GF7>::Advice(query);
    EXPECT_FALSE(ContainsSimpleSelector(expr.get()));
  }
}

TEST_F(SimpleSelectorFinderTest, Instance) {
  struct {
    int32_t rotation;
    size_t column_index;
  } tests[] = {
      {1, 1},
      {2, 2},
  };

  for (const auto& test : tests) {
    plonk::InstanceQuery query(1, Rotation(test.rotation),
                               plonk::InstanceColumnKey(test.column_index));
    Expr expr = ExpressionFactory<GF7>::Instance(query);
    EXPECT_FALSE(ContainsSimpleSelector(expr.get()));
  }
}

TEST_F(SimpleSelectorFinderTest, Challenges) {
  Expr expr =
      ExpressionFactory<GF7>::Challenge(plonk::Challenge(1, plonk::Phase(0)));
  EXPECT_FALSE(ContainsSimpleSelector(expr.get()));
}

TEST_F(SimpleSelectorFinderTest, Negated) {
  GF7 value = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Negated(ExpressionFactory<GF7>::Constant(value));
  EXPECT_FALSE(ContainsSimpleSelector(expr.get()));
}

TEST_F(SimpleSelectorFinderTest, Sum) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr = ExpressionFactory<GF7>::Sum(ExpressionFactory<GF7>::Constant(a),
                                          ExpressionFactory<GF7>::Constant(b));
  EXPECT_FALSE(ContainsSimpleSelector(expr.get()));
  expr = ExpressionFactory<GF7>::Sum(
      ExpressionFactory<GF7>::Constant(a),
      ExpressionFactory<GF7>::Selector(plonk::Selector::Simple(1)));
  EXPECT_TRUE(ContainsSimpleSelector(expr.get()));
}

TEST_F(SimpleSelectorFinderTest, Product) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr = ExpressionFactory<GF7>::Product(
      ExpressionFactory<GF7>::Constant(a), ExpressionFactory<GF7>::Constant(b));
  EXPECT_FALSE(ContainsSimpleSelector(expr.get()));
  expr = ExpressionFactory<GF7>::Product(
      ExpressionFactory<GF7>::Constant(a),
      ExpressionFactory<GF7>::Selector(plonk::Selector::Simple(1)));
  EXPECT_TRUE(ContainsSimpleSelector(expr.get()));
}

TEST_F(SimpleSelectorFinderTest, Scaled) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Scaled(ExpressionFactory<GF7>::Constant(a), b);
  EXPECT_FALSE(ContainsSimpleSelector(expr.get()));
  expr = ExpressionFactory<GF7>::Scaled(
      ExpressionFactory<GF7>::Selector(plonk::Selector::Simple(1)), GF7(3));
  EXPECT_TRUE(ContainsSimpleSelector(expr.get()));
}

}  // namespace tachyon::zk
