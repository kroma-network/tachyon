#include "tachyon/zk/r1cs/linear_combination.h"

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::zk::r1cs {

namespace {

using F = math::GF7;

class LinearCombinationTest : public testing::Test {
 public:
  static void SetUpTestSuite() { F::Init(); }
};

void TestDeduplicate(const std::vector<Term<F>>& terms,
                     const LinearCombination<F>& expected_lc) {
  LinearCombination<F> lc(terms);
  lc.Deduplicate();
  EXPECT_EQ(lc, expected_lc);
  EXPECT_EQ(LinearCombination<F>::CreateDeduplicated(terms), expected_lc);
  EXPECT_TRUE(lc.IsSorted());
}

// |lc| is copied on purpose.
template <typename T>
void TestAddition(LinearCombination<F> lc, const T& value,
                  const LinearCombination<F>& expected_lc) {
  EXPECT_EQ(lc + value, expected_lc);
  lc += value;
  EXPECT_EQ(lc, expected_lc);
}

// |lc| is copied on purpose.
template <typename T>
void TestSubtraction(LinearCombination<F> lc, const T& value,
                     const LinearCombination<F>& expected_lc) {
  EXPECT_EQ(lc - value, expected_lc);
  lc -= value;
  EXPECT_EQ(lc, expected_lc);
}

}  // namespace

TEST_F(LinearCombinationTest, Deduplicate) {
  Variable zero = Variable::Zero();
  Variable one = Variable::One();

  TestDeduplicate({}, LinearCombination<F>());
  TestDeduplicate({{F(2), zero}, {F(3), zero}},
                  LinearCombination<F>({{F(5), zero}}));
  TestDeduplicate(
      {{F(2), zero}, {F(3), one}, {F(2), one}, {F(2), zero}, {F(1), one}},
      LinearCombination<F>({{F(4), zero}, {F(6), one}}));
}

TEST_F(LinearCombinationTest, AdditionTerm) {
  Variable zero = Variable::Zero();
  Variable one = Variable::One();
  Variable instance = Variable::Instance(0);

  LinearCombination<F> lc({{F(5), zero}, {F(2), one}});
  {
    LinearCombination<F> expected_lc({{F(1), zero}, {F(2), one}});
    TestAddition(lc, Term<F>(F(3), zero), expected_lc);
  }
  {
    LinearCombination<F> expected_lc({{F(6), zero}, {F(2), one}});
    TestAddition(lc, zero, expected_lc);
  }

  {
    LinearCombination<F> expected_lc({{F(5), zero}, {F(5), one}});
    TestAddition(lc, Term<F>(F(3), one), expected_lc);
  }
  {
    LinearCombination<F> expected_lc({{F(5), zero}, {F(3), one}});
    TestAddition(lc, one, expected_lc);
  }

  {
    LinearCombination<F> expected_lc(
        {{F(5), zero}, {F(2), one}, {F(3), instance}});
    TestAddition(lc, Term<F>(F(3), instance), expected_lc);
  }
  {
    LinearCombination<F> expected_lc(
        {{F(5), zero}, {F(2), one}, {F(1), instance}});
    TestAddition(lc, instance, expected_lc);
  }
}

TEST_F(LinearCombinationTest, SubtractionTerm) {
  Variable zero = Variable::Zero();
  Variable one = Variable::One();
  Variable instance = Variable::Instance(0);

  LinearCombination<F> lc({{F(5), zero}, {F(2), one}});
  {
    LinearCombination<F> expected_lc({{F(2), zero}, {F(2), one}});
    TestSubtraction(lc, Term<F>(F(3), zero), expected_lc);
  }
  {
    LinearCombination<F> expected_lc({{F(4), zero}, {F(2), one}});
    TestSubtraction(lc, zero, expected_lc);
  }

  {
    LinearCombination<F> expected_lc({{F(5), zero}, {F(6), one}});
    TestSubtraction(lc, Term<F>(F(3), one), expected_lc);
  }
  {
    LinearCombination<F> expected_lc({{F(5), zero}, {F(1), one}});
    TestSubtraction(lc, one, expected_lc);
  }

  {
    LinearCombination<F> expected_lc(
        {{F(5), zero}, {F(2), one}, {F(4), instance}});
    TestSubtraction(lc, Term<F>(F(3), instance), expected_lc);
  }
  {
    LinearCombination<F> expected_lc(
        {{F(5), zero}, {F(2), one}, {F(6), instance}});
    TestSubtraction(lc, instance, expected_lc);
  }
}

TEST_F(LinearCombinationTest, ScalarMul) {
  Variable zero = Variable::Zero();
  Variable one = Variable::One();

  LinearCombination<F> lc({{F(5), zero}, {F(2), one}});
  LinearCombination<F> expected_lc({{F(3), zero}, {F(4), one}});
  EXPECT_EQ(lc * F(2), expected_lc);
  lc *= F(2);
  EXPECT_EQ(lc, expected_lc);
}

TEST_F(LinearCombinationTest, BinarySearch) {
  const size_t size = 10;
  std::vector<Term<F>> terms = base::CreateVector(
      size, [](size_t i) { return Term<F>(F(1), Variable::Instance(i << 1)); });
  LinearCombination<F> lc(std::move(terms));
  for (size_t i = 0; i < (size << 1); ++i) {
    size_t index;
    if (i % 2 == 0) {
      ASSERT_TRUE(lc.BinarySearch(Variable::Instance(i), &index));
      EXPECT_EQ(index, (i >> 1));
    } else {
      ASSERT_FALSE(lc.BinarySearch(Variable::Instance(i), &index));
      EXPECT_EQ(index, (i >> 1) + 1);
    }
  }
}

TEST_F(LinearCombinationTest, AdditionLinearCombination) {
  Variable zero = Variable::Zero();
  Variable one = Variable::One();
  Variable instance = Variable::Instance(0);

  LinearCombination<F> lc_a({{F(5), zero}, {F(2), instance}});
  {
    LinearCombination<F> lc_b({{F(1), zero}});
    LinearCombination<F> expected_lc({{F(6), zero}, {F(2), instance}});
    TestAddition(lc_a, lc_b, expected_lc);
  }

  {
    LinearCombination<F> lc_b({{F(1), zero}, {F(3), one}, {F(2), instance}});
    LinearCombination<F> expected_lc(
        {{F(6), zero}, {F(3), one}, {F(4), instance}});
    TestAddition(lc_a, lc_b, expected_lc);
  }
}

TEST_F(LinearCombinationTest, SubtractionLinearCombination) {
  Variable zero = Variable::Zero();
  Variable one = Variable::One();
  Variable instance = Variable::Instance(0);

  LinearCombination<F> lc_a({{F(5), zero}, {F(2), instance}});
  {
    LinearCombination<F> lc_b({{F(1), zero}});
    LinearCombination<F> expected_lc({{F(4), zero}, {F(2), instance}});
    TestSubtraction(lc_a, lc_b, expected_lc);
  }

  {
    LinearCombination<F> lc_b({{F(1), zero}, {F(3), one}, {F(2), instance}});
    LinearCombination<F> expected_lc(
        {{F(4), zero}, {F(4), one}, {F(0), instance}});
    TestSubtraction(lc_a, lc_b, expected_lc);
  }
}

}  // namespace tachyon::zk::r1cs
