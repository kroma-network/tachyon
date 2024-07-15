#include "tachyon/zk/air/constraint_system/constraint_system.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/zk/air/expressions/air_evaluator.h"
#include "tachyon/zk/air/expressions/expression_factory.h"

namespace tachyon::zk::air {

namespace {

using F = math::GF7;

class ConstraintSystemTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(ConstraintSystemTest, FibonacciAirTest) {
  ConstraintSystem<F> constraint_system(3, 2, 0);
  // Check initial inputs in the first row: public₀ = main₀,₀
  std::unique_ptr<Expression<F>> expr1 =
      ExpressionFactory<F>::Variable(Variable::Public(0)) -
      ExpressionFactory<F>::Variable(Variable::Main(0, 0));
  constraint_system.EnforceFirstRowConstraint(std::move(expr1));

  // Check initial inputs in the first row: public₁ = main₀,₁
  std::unique_ptr<Expression<F>> expr2 =
      ExpressionFactory<F>::Variable(Variable::Public(1)) -
      ExpressionFactory<F>::Variable(Variable::Main(0, 1));
  constraint_system.EnforceFirstRowConstraint(std::move(expr2));

  // Check transition step: main₁,₀  = main₀,₁
  std::unique_ptr<Expression<F>> expr3 =
      ExpressionFactory<F>::Variable(Variable::Main(1, 0)) -
      ExpressionFactory<F>::Variable(Variable::Main(0, 1));
  constraint_system.EnforceTransitionConstraint(std::move(expr3));

  // Check transition step: main₁,₁ = main₀,₀ + main₀,₁
  std::unique_ptr<Expression<F>> expr4 =
      ExpressionFactory<F>::Variable(Variable::Main(1, 1)) -
      ExpressionFactory<F>::Variable(Variable::Main(0, 0)) -
      ExpressionFactory<F>::Variable(Variable::Main(0, 1));
  constraint_system.EnforceTransitionConstraint(std::move(expr4));

  // In the last row, check if output is same: main₁,₁ = public₂
  std::unique_ptr<Expression<F>> expr5 =
      ExpressionFactory<F>::Variable(Variable::Main(1, 1)) -
      ExpressionFactory<F>::Variable(Variable::Public(2));
  constraint_system.EnforceLastRowConstraint(std::move(expr5));

  std::vector<F> public_values = {F(1), F(1), F(1)};
  math::RowMajorMatrix<F> main_trace(5, 2);

  // clang-format off
  main_trace << F(1), F(1),
                F(1), F(2),
                F(2), F(3),
                F(3), F(5),
                F(5), F(1);
  // clang-format on

  Trace<F> trace(std::move(main_trace));

  AirEvaluator<F, F> evaluator;
  bool is_satisfied =
      constraint_system.IsSatisfied(evaluator, public_values, trace);
  EXPECT_TRUE(is_satisfied);
  EXPECT_EQ(constraint_system.GetMaxConstraintDegree(), 2);
}

TEST_F(ConstraintSystemTest, PreprocessedAirTest) {
  ConstraintSystem<F> constraint_system(0, 2, 2);

  std::unique_ptr<Expression<F>> expr1 =
      ExpressionFactory<F>::Variable(Variable::Preprocessed(0, 0)) *
      (ExpressionFactory<F>::Variable(Variable::Main(0, 0)) -
       ExpressionFactory<F>::Variable(Variable::Main(1, 0)));
  constraint_system.EnforceTransitionConstraint(std::move(expr1));

  std::unique_ptr<Expression<F>> expr2 =
      ExpressionFactory<F>::Variable(Variable::Preprocessed(0, 1)) *
      (ExpressionFactory<F>::Variable(Variable::Main(0, 1)) -
       ExpressionFactory<F>::Variable(Variable::Main(1, 1)));
  constraint_system.EnforceTransitionConstraint(std::move(expr2));

  std::vector<F> public_values = {};
  math::RowMajorMatrix<F> main_trace(4, 2);
  math::RowMajorMatrix<F> preprocessed_trace(4, 2);

  // clang-format off
  main_trace << F(5), F(0),
                F(5), F(4),
                F(5), F(4),
                F(0), F(4);

  preprocessed_trace << F(1), F(0),
                        F(1), F(1),
                        F(0), F(1),
                        F(0), F(1);
  // clang-format on

  Trace<F> trace(std::move(main_trace), std::move(preprocessed_trace));

  AirEvaluator<F, F> evaluator;
  bool is_satisfied =
      constraint_system.IsSatisfied(evaluator, public_values, trace);
  EXPECT_TRUE(is_satisfied);
}

}  // namespace tachyon::zk::air
