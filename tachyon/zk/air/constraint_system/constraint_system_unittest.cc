#include "tachyon/zk/air/constraint_system/constraint_system.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/zk/air/expressions/evaluator.h"
#include "tachyon/zk/air/expressions/expression_factory.h"

namespace tachyon::zk::air {

namespace {

using F = math::GF7;

class ConstraintSystemTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(ConstraintSystemTest, FibonacciAirTest) {
  ConstraintSystem<F> constraint_system(0, 2, 3);
  const std::vector<Variable>& public_values =
      constraint_system.public_values();
  const std::vector<Variable>& main_values = constraint_system.main();

  // Check initial inputs in the first row: public₀ = main₀,₀
  std::unique_ptr<Expression<F>> expr1 =
      ExpressionFactory<F>::Variable(public_values[0]) -
      ExpressionFactory<F>::Variable(main_values[0]);
  constraint_system.EnforceFirstRowConstraint(std::move(expr1));

  // Check initial inputs in the first row: public₁ = main₀,₁
  std::unique_ptr<Expression<F>> expr2 =
      ExpressionFactory<F>::Variable(public_values[1]) -
      ExpressionFactory<F>::Variable(main_values[1]);
  constraint_system.EnforceFirstRowConstraint(std::move(expr2));

  // Check transition step: main₁,₀  = main₀,₁
  std::unique_ptr<Expression<F>> expr3 =
      ExpressionFactory<F>::Variable(main_values[1]) -
      ExpressionFactory<F>::Variable(main_values[2]);
  constraint_system.EnforceTransitionConstraint(std::move(expr3));

  // Check transition step: main₁,₁ = main₀,₀ + main₀,₁
  std::unique_ptr<Expression<F>> expr4 =
      ExpressionFactory<F>::Variable(main_values[3]) -
      ExpressionFactory<F>::Variable(main_values[0]) -
      ExpressionFactory<F>::Variable(main_values[1]);
  constraint_system.EnforceTransitionConstraint(std::move(expr4));

  // In the last row, check if output is same: main₁,₁ = public₂
  std::unique_ptr<Expression<F>> expr5 =
      ExpressionFactory<F>::Variable(main_values[3]) -
      ExpressionFactory<F>::Variable(public_values[2]);
  constraint_system.EnforceLastRowConstraint(std::move(expr5));

  std::vector<F> public_trace = {F(1), F(1), F(1)};
  math::RowMajorMatrix<F, 5, 2> main_trace;

  // clang-format off
    main_trace << F(1), F(1),
                  F(1), F(2),
                  F(2), F(3),
                  F(3), F(5),
                  F(5), F(1);
  // clang-format on

  Evaluator<F> evaluator;
  bool is_satisfied =
      constraint_system.IsSatisfied(evaluator, public_trace, main_trace);
  EXPECT_TRUE(is_satisfied);
}

TEST_F(ConstraintSystemTest, PreprocessedAirTest) {
  ConstraintSystem<F> constraint_system(2, 2, 0);
  const std::vector<Variable>& main_values = constraint_system.main();
  const std::vector<Variable>& preprocessed_values = constraint_system.main();

  std::unique_ptr<Expression<F>> expr1 =
      (ExpressionFactory<F>::Variable(preprocessed_values[0]) *
       (ExpressionFactory<F>::Variable(main_values[0]) -
        ExpressionFactory<F>::Variable(main_values[2]))) +
      (ExpressionFactory<F>::Constant(F::One()) -
       ExpressionFactory<F>::Variable(preprocessed_values[0])) *
          (ExpressionFactory<F>::Variable(main_values[1]) -
           ExpressionFactory<F>::Variable(main_values[3]));
  constraint_system.EnforceFirstRowConstraint(std::move(expr1));

  {
    std::vector<F> public_trace = {};
    math::RowMajorMatrix<F, 4, 2> main_trace;
    math::RowMajorMatrix<F, 4, 2> preprocessed_trace;

    // clang-format off
    main_trace << F(1), F(2),
                  F(1), F(3),
                  F(1), F(1),
                  F(3), F(1);

    preprocessed_trace << F(1), F(0),
                          F(1), F(0),
                          F(0), F(1),
                          F(0), F(1);
    // clang-format on

    Evaluator<F> evaluator;
    bool is_satisfied = constraint_system.IsSatisfied(
        evaluator, public_trace, preprocessed_trace, main_trace);
    EXPECT_TRUE(is_satisfied);
  }
}

}  // namespace tachyon::zk::air
