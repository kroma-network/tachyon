#include "tachyon/zk/r1cs/constraint_system/constraint_system.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk::r1cs {

namespace {

using F = math::GF7;

class ConstraintSystemTest : public math::FiniteFieldTest<F> {};

}  // namespace

TEST_F(ConstraintSystemTest, MakeRow) {
  ConstraintSystem<F> constraint_system;
  Variable v0 = constraint_system.CreateInstanceVariable([]() { return F(1); });
  Variable v1 = constraint_system.CreateInstanceVariable([]() { return F(2); });
  Variable v2 = constraint_system.CreateWitnessVariable([]() { return F(3); });
  Variable v3 = constraint_system.CreateWitnessVariable([]() { return F(4); });
  Variable v4 = constraint_system.CreateWitnessVariable([]() { return F(5); });
  LinearCombination<F> lc({
      {F(1), v0},
      {F(2), v1},
      {F(3), v2},
      {F(4), v3},
      {F(5), v4},
  });
  std::vector<Cell<F>> expected_row({
      {F(1), 1},
      {F(2), 2},
      {F(3), 3},
      {F(4), 4},
      {F(5), 5},
  });
  EXPECT_EQ(constraint_system.MakeRow(lc), expected_row);
}

TEST_F(ConstraintSystemTest, CreateZero) {
  ConstraintSystem<F> constraint_system;
  EXPECT_EQ(constraint_system.CreateZero(), Variable::Zero());
}

TEST_F(ConstraintSystemTest, CreateOne) {
  ConstraintSystem<F> constraint_system;
  EXPECT_EQ(constraint_system.CreateOne(), Variable::One());
}

TEST_F(ConstraintSystemTest, CreateInstanceVariable) {
  for (size_t i = 0; i < 2; ++i) {
    ConstraintSystem<F> constraint_system;
    if (i == 0) {
      constraint_system.set_mode(SynthesisMode::Setup());
    } else if (i == 1) {
      constraint_system.set_mode(SynthesisMode::Prove(true));
    } else {
      constraint_system.set_mode(SynthesisMode::Prove(false));
    }
    EXPECT_EQ(constraint_system.num_instance_variables(), 1);
    if (i == 0) {
      EXPECT_DEATH(
          constraint_system.CreateInstanceVariable([]() { return F(3); }), "");
    } else {
      Variable variable =
          constraint_system.CreateInstanceVariable([]() { return F(3); });
      EXPECT_EQ(variable, Variable::Instance(1));
      EXPECT_EQ(constraint_system.num_instance_variables(), 2);
    }
  }
}

TEST_F(ConstraintSystemTest, CreateWitnessVariable) {
  for (size_t i = 0; i < 3; ++i) {
    ConstraintSystem<F> constraint_system;
    if (i == 0) {
      constraint_system.set_mode(SynthesisMode::Setup());
    } else if (i == 1) {
      constraint_system.set_mode(SynthesisMode::Prove(true));
    } else {
      constraint_system.set_mode(SynthesisMode::Prove(false));
    }
    EXPECT_EQ(constraint_system.num_witness_variables(), 0);
    if (i == 0) {
      EXPECT_DEATH(
          constraint_system.CreateWitnessVariable([]() { return F(3); }), "");
    } else {
      Variable variable =
          constraint_system.CreateWitnessVariable([]() { return F(3); });
      EXPECT_EQ(variable, Variable::Witness(0));
      EXPECT_EQ(constraint_system.num_witness_variables(), 1);
    }
  }
}

TEST_F(ConstraintSystemTest, CreateLinearCombination) {
  ConstraintSystem<F> constraint_system;
  EXPECT_EQ(constraint_system.num_linear_combinations(), 0);
  EXPECT_EQ(constraint_system.CreateLinearCombination(LinearCombination<F>()),
            Variable::SymbolicLinearCombination(0));
  EXPECT_EQ(constraint_system.num_linear_combinations(), 1);
}

TEST_F(ConstraintSystemTest, EnforceConstraint) {
  for (size_t i = 0; i < 3; ++i) {
    ConstraintSystem<F> constraint_system;
    if (i == 0) {
      constraint_system.set_mode(SynthesisMode::Setup());
    } else if (i == 1) {
      constraint_system.set_mode(SynthesisMode::Prove(true));
    } else {
      constraint_system.set_mode(SynthesisMode::Prove(false));
    }
    EXPECT_EQ(constraint_system.num_constraints(), 0);
    EXPECT_EQ(constraint_system.num_linear_combinations(), 0);
    if (i < 2) {
      constraint_system.EnforceConstraint(LinearCombination<F>(),
                                          LinearCombination<F>(),
                                          LinearCombination<F>());
      EXPECT_EQ(constraint_system.num_constraints(), 1);
      EXPECT_EQ(constraint_system.num_linear_combinations(), 3);
    } else {
      EXPECT_DEATH(constraint_system.EnforceConstraint(LinearCombination<F>(),
                                                       LinearCombination<F>(),
                                                       LinearCombination<F>()),
                   "");
    }
  }
}

TEST_F(ConstraintSystemTest, ComputeLCNumTimesUsed) {
  ConstraintSystem<F> constraint_system;
  Variable lc_variable =
      constraint_system.CreateLinearCombination(LinearCombination<F>());
  LinearCombination<F> lc;
  lc += Term<F>(lc_variable);
  lc += Term<F>(lc_variable);
  Variable lc_variable2 = constraint_system.CreateLinearCombination(lc);
  LinearCombination<F> lc2;
  lc2 += Term<F>(lc_variable);
  lc2 += Term<F>(lc_variable2);
  constraint_system.CreateLinearCombination(lc2);
  std::vector<size_t> expected_lc_times_used = {2, 1, 0};
  EXPECT_EQ(constraint_system.ComputeLCNumTimesUsed(false),
            expected_lc_times_used);
  expected_lc_times_used = {3, 2, 1};
  EXPECT_EQ(constraint_system.ComputeLCNumTimesUsed(true),
            expected_lc_times_used);
}

TEST_F(ConstraintSystemTest, Finalize) {
  for (size_t i = 0; i < 2; ++i) {
    ConstraintSystem<F> constraint_system;
    if (i == 1) {
      constraint_system.set_optimization_goal(OptimizationGoal::kWeight);
    }
    // a = 1
    Variable a =
        constraint_system.CreateInstanceVariable([]() { return F(1); });
    // b = 1
    Variable b = constraint_system.CreateWitnessVariable([]() { return F(1); });
    // c = 2
    Variable c = constraint_system.CreateWitnessVariable([]() { return F(2); });
    // a * 2b = c
    constraint_system.EnforceConstraint(LinearCombination<F>({{F(1), a}}),
                                        LinearCombination<F>({{F(2), b}}),
                                        LinearCombination<F>({{F(1), c}}));
    // d = a + b
    Variable d = constraint_system.CreateLinearCombination(
        LinearCombination<F>({{F(1), a}, {F(1), b}}));
    // a * d = d
    constraint_system.EnforceConstraint(LinearCombination<F>({{F(1), a}}),
                                        LinearCombination<F>({{F(1), d}}),
                                        LinearCombination<F>({{F(1), d}}));
    // e = d + d
    Variable e = constraint_system.CreateLinearCombination(
        LinearCombination<F>({{F(1), d}, {F(1), d}}));
    // 1 * e = e
    constraint_system.EnforceConstraint(
        LinearCombination<F>({{F(1), Variable::One()}}),
        LinearCombination<F>({{F(1), e}}), LinearCombination<F>({{F(1), e}}));

    constraint_system.Finalize();

    ConstraintMatrices<F> matrices = constraint_system.ToMatrices().value();
    ConstraintMatrices<F> expected_matrices;
    if (i == 0) {
      expected_matrices.num_instance_variables = 2;
      expected_matrices.num_witness_variables = 2;
      expected_matrices.num_constraints = 3;
      expected_matrices.a_num_non_zero = 3;
      expected_matrices.b_num_non_zero = 5;
      expected_matrices.c_num_non_zero = 5;
      // a * 2b  = c
      // a * (a + b) = a + b
      // 1 * (2a + 2b) = 2a + 2b
      expected_matrices.a = Matrix<F>({{{F(1), 1}}, {{F(1), 1}}, {{F(1), 0}}});
      expected_matrices.b = Matrix<F>(
          {{{F(2), 2}}, {{F(1), 1}, {F(1), 2}}, {{F(2), 1}, {F(2), 2}}});
      expected_matrices.c = Matrix<F>(
          {{{F(1), 3}}, {{F(1), 1}, {F(1), 2}}, {{F(2), 1}, {F(2), 2}}});
    } else {
      expected_matrices.num_instance_variables = 2;
      expected_matrices.num_witness_variables = 3;
      expected_matrices.num_constraints = 4;
      expected_matrices.a_num_non_zero = 5;
      expected_matrices.b_num_non_zero = 4;
      expected_matrices.c_num_non_zero = 4;
      // a * 2b  = c
      // a * d = d
      // 1 * 2d = 2d
      // (a + b) * 1 = d
      expected_matrices.a = Matrix<F>(
          {{{F(1), 1}}, {{F(1), 1}}, {{F(1), 0}}, {{F(1), 1}, {F(1), 2}}});
      expected_matrices.b =
          Matrix<F>({{{F(2), 2}}, {{F(1), 4}}, {{F(2), 4}}, {{F(1), 0}}});
      expected_matrices.c =
          Matrix<F>({{{F(1), 3}}, {{F(1), 4}}, {{F(2), 4}}, {{F(1), 4}}});
    }
    EXPECT_EQ(matrices, expected_matrices);
  }
}

TEST_F(ConstraintSystemTest, EvalLinearCombination) {
  ConstraintSystem<F> constraint_system;
  Variable instance_variable =
      constraint_system.CreateInstanceVariable([]() { return F(2); });
  Variable advice_variable =
      constraint_system.CreateWitnessVariable([]() { return F(3); });
  LinearCombination<F> lc;
  lc += Term<F>(F(4), instance_variable);  // 4 * 2 = 1
  lc += Term<F>(F(5), advice_variable);    // 5 * 3 = 1
  Variable lc_variable =
      constraint_system.CreateLinearCombination(std::move(lc));
  EXPECT_EQ(constraint_system.EvalLinearCombination(lc_variable.index()), F(2));
}

TEST_F(ConstraintSystemTest, GetAssignedValue) {
  ConstraintSystem<F> constraint_system;
  EXPECT_EQ(constraint_system.GetAssignedValue(Variable::Zero()), F(0));
  EXPECT_EQ(constraint_system.GetAssignedValue(Variable::One()), F(1));
  Variable instance_variable =
      constraint_system.CreateInstanceVariable([]() { return F(2); });
  EXPECT_EQ(constraint_system.GetAssignedValue(instance_variable), F(2));
  Variable advice_variable =
      constraint_system.CreateWitnessVariable([]() { return F(3); });
  EXPECT_EQ(constraint_system.GetAssignedValue(advice_variable), F(3));
  LinearCombination<F> lc;
  lc += Term<F>(F(4), instance_variable);  // 4 * 2 = 1
  lc += Term<F>(F(5), advice_variable);    // 5 * 3 = 1
  Variable lc_variable =
      constraint_system.CreateLinearCombination(std::move(lc));
  EXPECT_EQ(constraint_system.GetAssignedValue(lc_variable), F(2));
  EXPECT_EQ(constraint_system.lc_assignment_cache_[lc_variable.index()], F(2));
}

TEST_F(ConstraintSystemTest, IsSatisfied) {
  ConstraintSystem<F> constraint_system;
  LinearCombination<F> a({{F(2), Variable::One()}});
  LinearCombination<F> b({{F(3), Variable::One()}});
  LinearCombination<F> c({{F(6), Variable::One()}});
  constraint_system.EnforceConstraint(a, b, c);
  ASSERT_TRUE(constraint_system.IsSatisfied());
  a = LinearCombination<F>({{F(2), Variable::One()}});
  b = LinearCombination<F>({{F(3), Variable::One()}});
  c = LinearCombination<F>({{F(5), Variable::One()}});
  constraint_system.EnforceConstraint(a, b, c);
  ASSERT_TRUE(!constraint_system.IsSatisfied());
}

}  // namespace tachyon::zk::r1cs
