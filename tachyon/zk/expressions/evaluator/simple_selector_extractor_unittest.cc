#include "tachyon/zk/expressions/evaluator/simple_selector_extractor.h"

#include <memory>

#include "tachyon/zk/expressions/evaluator/test/evaluator_test.h"
#include "tachyon/zk/expressions/expression_factory.h"

namespace tachyon::zk {

using Expr = std::unique_ptr<Expression<GF7>>;

class SimpleSelectorExtractorTest : public EvaluatorTest {};

TEST_F(SimpleSelectorExtractorTest, Constant) {
  GF7 value = GF7::Random();
  std::unique_ptr<Expression<GF7>> expr =
      ExpressionFactory<GF7>::Constant(value);
  EXPECT_FALSE(ExtractSimpleSelector(expr.get()).has_value());
}

TEST_F(SimpleSelectorExtractorTest, Selector) {
  plonk::Selector expected_selector = plonk::Selector::Simple(1);
  Expr expr = ExpressionFactory<GF7>::Selector(expected_selector);
  std::optional<plonk::Selector> selector = ExtractSimpleSelector(expr.get());
  EXPECT_EQ(selector.value(), expected_selector);
  expr = ExpressionFactory<GF7>::Selector(plonk::Selector::Complex(1));
  EXPECT_FALSE(ExtractSimpleSelector(expr.get()).has_value());
}

TEST_F(SimpleSelectorExtractorTest, Fixed) {
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
    EXPECT_FALSE(ExtractSimpleSelector(expr.get()).has_value());
  }
}

TEST_F(SimpleSelectorExtractorTest, Advice) {
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
    EXPECT_FALSE(ExtractSimpleSelector(expr.get()).has_value());
  }
}

TEST_F(SimpleSelectorExtractorTest, Instance) {
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
    EXPECT_FALSE(ExtractSimpleSelector(expr.get()).has_value());
  }
}

TEST_F(SimpleSelectorExtractorTest, Challenges) {
  Expr expr =
      ExpressionFactory<GF7>::Challenge(plonk::Challenge(1, plonk::Phase(0)));
  EXPECT_FALSE(ExtractSimpleSelector(expr.get()).has_value());
}

TEST_F(SimpleSelectorExtractorTest, Negated) {
  GF7 value = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Negated(ExpressionFactory<GF7>::Constant(value));
  EXPECT_FALSE(ExtractSimpleSelector(expr.get()).has_value());
}

TEST_F(SimpleSelectorExtractorTest, Sum) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr = ExpressionFactory<GF7>::Sum(ExpressionFactory<GF7>::Constant(a),
                                          ExpressionFactory<GF7>::Constant(b));
  EXPECT_FALSE(ExtractSimpleSelector(expr.get()).has_value());
  plonk::Selector expected_selector = plonk::Selector::Simple(1);
  expr = ExpressionFactory<GF7>::Sum(
      ExpressionFactory<GF7>::Constant(a),
      ExpressionFactory<GF7>::Selector(expected_selector));
  std::optional<plonk::Selector> selector = ExtractSimpleSelector(expr.get());
  EXPECT_EQ(selector.value(), expected_selector);
}

TEST_F(SimpleSelectorExtractorTest, Product) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr = ExpressionFactory<GF7>::Product(
      ExpressionFactory<GF7>::Constant(a), ExpressionFactory<GF7>::Constant(b));
  EXPECT_FALSE(ExtractSimpleSelector(expr.get()).has_value());
  plonk::Selector expected_selector = plonk::Selector::Simple(1);
  expr = ExpressionFactory<GF7>::Product(
      ExpressionFactory<GF7>::Constant(a),
      ExpressionFactory<GF7>::Selector(expected_selector));
  std::optional<plonk::Selector> selector = ExtractSimpleSelector(expr.get());
  EXPECT_EQ(selector.value(), expected_selector);
}

TEST_F(SimpleSelectorExtractorTest, Scaled) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Scaled(ExpressionFactory<GF7>::Constant(a), b);
  EXPECT_FALSE(ExtractSimpleSelector(expr.get()).has_value());
  plonk::Selector expected_selector = plonk::Selector::Simple(1);
  expr = ExpressionFactory<GF7>::Scaled(
      ExpressionFactory<GF7>::Selector(expected_selector), GF7(3));
  std::optional<plonk::Selector> selector = ExtractSimpleSelector(expr.get());
  EXPECT_EQ(selector.value(), expected_selector);
}

}  // namespace tachyon::zk
