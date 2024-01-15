#include "tachyon/zk/r1cs/variable.h"

#include "gtest/gtest.h"

#include "tachyon/base/random.h"

namespace tachyon::zk::r1cs {

TEST(VariableTest, Comparison) {
  size_t a = base::Uniform(base::Range<size_t>());
  size_t b = base::Uniform(base::Range<size_t>());

  EXPECT_EQ(Variable::Zero(), Variable::Zero());
  EXPECT_LT(Variable::Zero(), Variable::One());
  EXPECT_LT(Variable::Zero(), Variable::Instance(a));
  EXPECT_LT(Variable::Zero(), Variable::Witness(a));
  EXPECT_LT(Variable::Zero(), Variable::SymbolicLinearCombination(a));

  EXPECT_GT(Variable::One(), Variable::Zero());
  EXPECT_EQ(Variable::One(), Variable::One());
  EXPECT_LT(Variable::One(), Variable::Instance(a));
  EXPECT_LT(Variable::One(), Variable::Witness(a));
  EXPECT_LT(Variable::One(), Variable::SymbolicLinearCombination(a));

  EXPECT_GT(Variable::Instance(a), Variable::Zero());
  EXPECT_GT(Variable::Instance(a), Variable::One());
  EXPECT_EQ(Variable::Instance(a), Variable::Instance(a));
  EXPECT_EQ(Variable::Instance(a) < Variable::Instance(b), a < b);
  EXPECT_LT(Variable::Instance(a), Variable::Witness(a));
  EXPECT_LT(Variable::Instance(a), Variable::SymbolicLinearCombination(a));

  EXPECT_GT(Variable::Witness(a), Variable::Zero());
  EXPECT_GT(Variable::Witness(a), Variable::One());
  EXPECT_GT(Variable::Witness(a), Variable::Instance(a));
  EXPECT_EQ(Variable::Witness(a), Variable::Witness(a));
  EXPECT_EQ(Variable::Witness(a) < Variable::Witness(b), a < b);
  EXPECT_LT(Variable::Witness(a), Variable::SymbolicLinearCombination(a));

  EXPECT_GT(Variable::SymbolicLinearCombination(a), Variable::Zero());
  EXPECT_GT(Variable::SymbolicLinearCombination(a), Variable::One());
  EXPECT_GT(Variable::SymbolicLinearCombination(a), Variable::Instance(a));
  EXPECT_GT(Variable::SymbolicLinearCombination(a), Variable::Witness(a));
  EXPECT_EQ(Variable::SymbolicLinearCombination(a),
            Variable::SymbolicLinearCombination(a));
  EXPECT_EQ(Variable::SymbolicLinearCombination(a) <
                Variable::SymbolicLinearCombination(b),
            a < b);
}

}  // namespace tachyon::zk::r1cs
