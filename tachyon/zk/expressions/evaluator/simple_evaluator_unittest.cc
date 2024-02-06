#include "tachyon/zk/expressions/evaluator/simple_evaluator.h"

#include <memory>

#include "tachyon/zk/expressions/evaluator/test/evaluator_test.h"
#include "tachyon/zk/expressions/expression_factory.h"

namespace tachyon::zk {

namespace {

using Expr = std::unique_ptr<Expression<GF7>>;

class SimpleEvaluatorTest : public EvaluatorTest {
 public:
  void SetUp() override {
    std::vector<GF7> evaluations;

    for (size_t i = 0; i < 5; ++i) {
      evaluations =
          base::CreateVector(kMaxDegree + 1, []() { return GF7::Random(); });
      fixed_columns_.push_back(Evals(evaluations));
      advice_columns_.push_back(Evals(evaluations));
      instance_columns_.push_back(Evals(evaluations));

      challenges_.push_back(GF7::Random());
    }

    plonk::RefTable<Evals> columns(absl::MakeConstSpan(fixed_columns_),
                                   absl::MakeConstSpan(advice_columns_),
                                   absl::MakeConstSpan(instance_columns_));
    simple_evaluator_ = std::make_unique<SimpleEvaluator<Evals>>(
        3, 4, 1, columns, absl::MakeConstSpan(challenges_));
  }

 protected:
  std::unique_ptr<SimpleEvaluator<Evals>> simple_evaluator_;
};

}  // namespace

TEST_F(SimpleEvaluatorTest, Constant) {
  GF7 value = GF7::Random();
  Expr expr = ExpressionFactory<GF7>::Constant(value);
  GF7 evaluated = simple_evaluator_->Evaluate(expr.get());
  EXPECT_EQ(value, evaluated);
}

TEST_F(SimpleEvaluatorTest, Selector) {
  Expr expr = ExpressionFactory<GF7>::Selector(plonk::Selector::Simple(1));
  EXPECT_DEATH(simple_evaluator_->Evaluate(expr.get()), "");
}

TEST_F(SimpleEvaluatorTest, Fixed) {
  struct {
    int32_t rotation;
    size_t column_index;
  } tests[] = {
      // fixed_columns_[0], (3 + 1 * 1) % 4 = 0, evaluations[0]
      {1, 0},
      // fixed_columns_[1], (4 + 2 * 1) % 4 = 2, evaluations[2]
      {2, 1},
  };

  for (const auto& test : tests) {
    int32_t idx = simple_evaluator_->idx();
    int32_t rot_scale = simple_evaluator_->rot_scale();
    int32_t size = simple_evaluator_->size();
    plonk::FixedQuery query(1, Rotation(test.rotation),
                            plonk::FixedColumnKey(test.column_index));
    RowIndex row_index = query.rotation().GetIndex(idx, rot_scale, size);

    const GF7& expected = fixed_columns_[test.column_index][row_index];
    Expr expr = ExpressionFactory<GF7>::Fixed(query);
    GF7 evaluated = simple_evaluator_->Evaluate(expr.get());
    EXPECT_EQ(evaluated, expected);
  }
}

TEST_F(SimpleEvaluatorTest, Advice) {
  struct {
    int32_t rotation;
    size_t column_index;
  } tests[] = {
      // advice_columns_[2], (3 + 6 * 1) % 4 = 1, evaluations[1]
      {6, 2},
      // advice_columns_[3], (4 + 7 * 1) % 4 = 3, evaluations[3]
      {7, 3},
  };

  for (const auto& test : tests) {
    int32_t idx = simple_evaluator_->idx();
    int32_t rot_scale = simple_evaluator_->rot_scale();
    int32_t size = simple_evaluator_->size();
    plonk::AdviceQuery query(
        1, Rotation(test.rotation),
        plonk::AdviceColumnKey(test.column_index, plonk::Phase(0)));
    RowIndex row_index = query.rotation().GetIndex(idx, rot_scale, size);

    const GF7& expected = advice_columns_[test.column_index][row_index];
    Expr expr = ExpressionFactory<GF7>::Advice(query);
    GF7 evaluated = simple_evaluator_->Evaluate(expr.get());
    EXPECT_EQ(evaluated, expected);
  }
}

TEST_F(SimpleEvaluatorTest, Instance) {
  struct {
    int32_t rotation;
    size_t column_index;
  } tests[] = {
      // instance_columns_[1], (3 + 1 * 1) % 4 = 0, evaluations[0]
      {1, 1},
      // instance_columns_[2], (4 + 2 * 1) % 4 = 2, evaluations[2]
      {2, 2},
  };

  for (const auto& test : tests) {
    int32_t idx = simple_evaluator_->idx();
    int32_t rot_scale = simple_evaluator_->rot_scale();
    int32_t size = simple_evaluator_->size();
    plonk::InstanceQuery query(1, Rotation(test.rotation),
                               plonk::InstanceColumnKey(test.column_index));
    RowIndex row_index = query.rotation().GetIndex(idx, rot_scale, size);

    const GF7& expected = instance_columns_[test.column_index][row_index];
    Expr expr = ExpressionFactory<GF7>::Instance(query);
    GF7 evaluated = simple_evaluator_->Evaluate(expr.get());
    EXPECT_EQ(evaluated, expected);
  }
}

TEST_F(SimpleEvaluatorTest, Challenges) {
  for (size_t i = 0; i < challenges_.size(); ++i) {
    Expr expr =
        ExpressionFactory<GF7>::Challenge(plonk::Challenge(i, plonk::Phase(0)));
    GF7 evaluated = simple_evaluator_->Evaluate(expr.get());
    GF7 expected = challenges_[i];
    EXPECT_EQ(evaluated, expected);
  }
}

TEST_F(SimpleEvaluatorTest, Negated) {
  GF7 value = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Negated(ExpressionFactory<GF7>::Constant(value));
  GF7 negated = simple_evaluator_->Evaluate(expr.get());
  EXPECT_EQ(negated, -value);
}

TEST_F(SimpleEvaluatorTest, Sum) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr = ExpressionFactory<GF7>::Sum(ExpressionFactory<GF7>::Constant(a),
                                          ExpressionFactory<GF7>::Constant(b));
  GF7 evaluated = simple_evaluator_->Evaluate(expr.get());
  EXPECT_EQ(a + b, evaluated);
}

TEST_F(SimpleEvaluatorTest, Product) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr = ExpressionFactory<GF7>::Product(
      ExpressionFactory<GF7>::Constant(a), ExpressionFactory<GF7>::Constant(b));
  GF7 evaluated = simple_evaluator_->Evaluate(expr.get());
  EXPECT_EQ(a * b, evaluated);
}

TEST_F(SimpleEvaluatorTest, Scaled) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Scaled(ExpressionFactory<GF7>::Constant(a), b);
  GF7 scaled_expr = simple_evaluator_->Evaluate(expr.get());
  EXPECT_EQ(scaled_expr, a * b);
}

}  // namespace tachyon::zk
