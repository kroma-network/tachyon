#include "tachyon/zk/plonk/expressions/proving_evaluator.h"

#include <memory>

#include "tachyon/zk/plonk/expressions/evaluator/test/evaluator_test.h"
#include "tachyon/zk/plonk/expressions/expression_factory.h"

namespace tachyon::zk::plonk {

namespace {

constexpr size_t kMaxDegree = 5;

using GF7 = math::GF7;
using Evals = math::UnivariateEvaluations<GF7, kMaxDegree>;
using Expr = std::unique_ptr<Expression<GF7>>;

class ProvingEvaluatorTest : public EvaluatorTest {
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

    MultiPhaseRefTable<Evals> table(fixed_columns_, advice_columns_,
                                    instance_columns_, challenges_);
    proving_evaluator_ =
        std::make_unique<ProvingEvaluator<Evals>>(3, 4, 1, table);
  }

 protected:
  std::unique_ptr<ProvingEvaluator<Evals>> proving_evaluator_;
};

}  // namespace

TEST_F(ProvingEvaluatorTest, Constant) {
  GF7 value = GF7::Random();
  Expr expr = ExpressionFactory<GF7>::Constant(value);
  GF7 evaluated = proving_evaluator_->Evaluate(expr.get());
  EXPECT_EQ(value, evaluated);
}

TEST_F(ProvingEvaluatorTest, Selector) {
  Expr expr = ExpressionFactory<GF7>::Selector(Selector::Simple(1));
  EXPECT_DEATH(proving_evaluator_->Evaluate(expr.get()), "");
}

TEST_F(ProvingEvaluatorTest, Fixed) {
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
    int32_t idx = proving_evaluator_->idx();
    int32_t rot_scale = proving_evaluator_->rot_scale();
    int32_t size = proving_evaluator_->size();
    FixedQuery query(1, Rotation(test.rotation),
                     FixedColumnKey(test.column_index));
    RowIndex row_index = query.rotation().GetIndex(idx, rot_scale, size);

    const GF7& expected = fixed_columns_[test.column_index][row_index];
    Expr expr = ExpressionFactory<GF7>::Fixed(query);
    GF7 evaluated = proving_evaluator_->Evaluate(expr.get());
    EXPECT_EQ(evaluated, expected);
  }
}

TEST_F(ProvingEvaluatorTest, Advice) {
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
    int32_t idx = proving_evaluator_->idx();
    int32_t rot_scale = proving_evaluator_->rot_scale();
    int32_t size = proving_evaluator_->size();
    AdviceQuery query(1, Rotation(test.rotation),
                      AdviceColumnKey(test.column_index, Phase(0)));
    RowIndex row_index = query.rotation().GetIndex(idx, rot_scale, size);

    const GF7& expected = advice_columns_[test.column_index][row_index];
    Expr expr = ExpressionFactory<GF7>::Advice(query);
    GF7 evaluated = proving_evaluator_->Evaluate(expr.get());
    EXPECT_EQ(evaluated, expected);
  }
}

TEST_F(ProvingEvaluatorTest, Instance) {
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
    int32_t idx = proving_evaluator_->idx();
    int32_t rot_scale = proving_evaluator_->rot_scale();
    int32_t size = proving_evaluator_->size();
    InstanceQuery query(1, Rotation(test.rotation),
                        InstanceColumnKey(test.column_index));
    RowIndex row_index = query.rotation().GetIndex(idx, rot_scale, size);

    const GF7& expected = instance_columns_[test.column_index][row_index];
    Expr expr = ExpressionFactory<GF7>::Instance(query);
    GF7 evaluated = proving_evaluator_->Evaluate(expr.get());
    EXPECT_EQ(evaluated, expected);
  }
}

TEST_F(ProvingEvaluatorTest, Challenges) {
  for (size_t i = 0; i < challenges_.size(); ++i) {
    Expr expr = ExpressionFactory<GF7>::Challenge(Challenge(i, Phase(0)));
    GF7 evaluated = proving_evaluator_->Evaluate(expr.get());
    GF7 expected = challenges_[i];
    EXPECT_EQ(evaluated, expected);
  }
}

TEST_F(ProvingEvaluatorTest, Negated) {
  GF7 value = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Negated(ExpressionFactory<GF7>::Constant(value));
  GF7 negated = proving_evaluator_->Evaluate(expr.get());
  EXPECT_EQ(negated, -value);
}

TEST_F(ProvingEvaluatorTest, Sum) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr = ExpressionFactory<GF7>::Sum(ExpressionFactory<GF7>::Constant(a),
                                          ExpressionFactory<GF7>::Constant(b));
  GF7 evaluated = proving_evaluator_->Evaluate(expr.get());
  EXPECT_EQ(a + b, evaluated);
}

TEST_F(ProvingEvaluatorTest, Product) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr = ExpressionFactory<GF7>::Product(
      ExpressionFactory<GF7>::Constant(a), ExpressionFactory<GF7>::Constant(b));
  GF7 evaluated = proving_evaluator_->Evaluate(expr.get());
  EXPECT_EQ(a * b, evaluated);
}

TEST_F(ProvingEvaluatorTest, Scaled) {
  GF7 a = GF7::Random();
  GF7 b = GF7::Random();
  Expr expr =
      ExpressionFactory<GF7>::Scaled(ExpressionFactory<GF7>::Constant(a), b);
  GF7 scaled_expr = proving_evaluator_->Evaluate(expr.get());
  EXPECT_EQ(scaled_expr, a * b);
}

}  // namespace tachyon::zk::plonk
