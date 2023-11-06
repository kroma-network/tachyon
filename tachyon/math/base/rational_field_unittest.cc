#include "tachyon/math/base/rational_field.h"

#include <vector>

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

using R = RationalField<GF7>;

namespace {

class RationalFieldTest : public testing::Test {
 public:
  static void SetUpTestSuite() { GF7::Init(); }
};

}  // namespace

TEST_F(RationalFieldTest, Zero) {
  EXPECT_TRUE(R::Zero().IsZero());
  EXPECT_FALSE(R::One().IsZero());
}

TEST_F(RationalFieldTest, One) {
  EXPECT_TRUE(R::Zero().IsZero());
  EXPECT_FALSE(R::One().IsZero());
  EXPECT_FALSE(R(GF7(3), GF7(3)).IsZero());
}

TEST_F(RationalFieldTest, Random) {
  bool success = false;
  R r = R::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != R::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(RationalFieldTest, NumeratorOnlyComparison) {
  GF7 a_numerator = GF7::Random();
  GF7 b_numerator = GF7::Random();
  R a(a_numerator);
  R b(b_numerator);

#define COMPARISON_TEST(op)           \
  do {                                \
    if (a_numerator op b_numerator) { \
      EXPECT_TRUE(a op b);            \
    }                                 \
  } while (false)

  COMPARISON_TEST(==);
  COMPARISON_TEST(!=);
  COMPARISON_TEST(<);  // NOLINT(whitespace/operators)
  COMPARISON_TEST(<=);
  COMPARISON_TEST(>);  // NOLINT(whitespace/operators)
  COMPARISON_TEST(>=);

#undef COMPARISON_TEST
}

TEST_F(RationalFieldTest, Comparison) {
  R a = R::Random();
  R b = R::Random();

#define COMPARISON_TEST(op)                        \
  do {                                             \
    if ((a.numerator() * b.denominator())          \
            op(b.numerator() * a.denominator())) { \
      EXPECT_TRUE(a op b);                         \
    }                                              \
  } while (false)

  COMPARISON_TEST(==);
  COMPARISON_TEST(!=);
  COMPARISON_TEST(<);  // NOLINT(whitespace/operators)
  COMPARISON_TEST(<=);
  COMPARISON_TEST(>);  // NOLINT(whitespace/operators)
  COMPARISON_TEST(>=);

#undef COMPARISON_TEST
}

TEST_F(RationalFieldTest, MpzClassConversion) {
  R a = R::Random();
  EXPECT_EQ(a.Evaluate(), a.numerator() / a.denominator());
}

TEST_F(RationalFieldTest, AdditiveOperators) {
  struct {
    R a;
    R b;
    R sum;
    R amb;
    R bma;
  } tests[] = {
      {R(GF7(3), GF7(2)), R(GF7(2), GF7(5)), R(GF7(5), GF7(3)),
       R(GF7(4), GF7(3)), R(GF7(3), GF7(3))},
      {R(GF7(5), GF7(3)), R(GF7(1), GF7(2)), R(GF7(6), GF7(6)),
       R(GF7(0), GF7(6)), R(GF7(0), GF7(6))},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);

    R tmp = test.a;
    tmp += test.b;
    EXPECT_EQ(tmp, test.sum);
    tmp -= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(RationalFieldTest, AdditiveGroupOperators) {
  R r(GF7(3), GF7(2));
  R neg_expected(GF7(4), GF7(2));
  EXPECT_EQ(-r, neg_expected);
  r.NegInPlace();
  EXPECT_EQ(r, neg_expected);

  r = R(GF7(3), GF7(2));
  R dbl_expected(GF7(6), GF7(2));
  EXPECT_EQ(r.Double(), dbl_expected);
  r.DoubleInPlace();
  EXPECT_EQ(r, dbl_expected);
}

TEST_F(RationalFieldTest, MultiplicativeOperators) {
  struct {
    R a;
    R b;
    R mul;
    R adb;
    R bda;
  } tests[] = {
      {R(GF7(3), GF7(2)), R(GF7(2), GF7(5)), R(GF7(6), GF7(3)),
       R(GF7(1), GF7(4)), R(GF7(4), GF7(1))},
      {R(GF7(5), GF7(3)), R(GF7(1), GF7(2)), R(GF7(5), GF7(6)),
       R(GF7(3), GF7(3)), R(GF7(3), GF7(3))},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a * test.b, test.mul);
    EXPECT_EQ(test.b * test.a, test.mul);
    EXPECT_EQ(test.a / test.b, test.adb);
    EXPECT_EQ(test.b / test.a, test.bda);

    R tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    tmp /= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(RationalFieldTest, MultiplicativeGroupOperators) {
  R r = R::Random();
  EXPECT_TRUE((r * r.Inverse()).IsOne());
  R r_tmp = r;
  r.InverseInPlace();
  EXPECT_TRUE((r * r_tmp).IsOne());

  r = R(GF7(3), GF7(2));
  R expected = R(GF7(2), GF7(4));
  EXPECT_EQ(r.Square(), expected);
  r.SquareInPlace();
  EXPECT_EQ(r, expected);

  r = R(GF7(3), GF7(2));
  EXPECT_EQ(r.Pow(5), R(GF7(5), GF7(4)));
}

TEST_F(RationalFieldTest, BatchEvaluate) {
  size_t size = 100;
  std::vector<R> test_set =
      base::CreateVector(size, []() { return R::Random(); });
  std::vector<GF7> results;
  results.resize(100);
  ASSERT_TRUE(R::BatchEvaluate(test_set, &results));
  for (size_t i = 0; i < results.size(); ++i) {
    EXPECT_EQ(test_set[i].Evaluate(), results[i]);
  }
}

}  // namespace tachyon::math
