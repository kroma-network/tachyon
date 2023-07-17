#include "tachyon/math/finite_fields/prime_field.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

namespace {

class PrimeFieldTest : public ::testing::Test {
 public:
  PrimeFieldTest() { GF7Config::Init(); }
  PrimeFieldTest(const PrimeFieldTest&) = delete;
  PrimeFieldTest& operator=(const PrimeFieldTest&) = delete;
  ~PrimeFieldTest() override = default;
};

}  // namespace

TEST_F(PrimeFieldTest, FromString) {
  EXPECT_EQ(GF7::FromDecString("3"), GF7(3));
  EXPECT_EQ(GF7::FromHexString("0x3"), GF7(3));
}

TEST_F(PrimeFieldTest, ToString) {
  GF7 f(3);

  EXPECT_EQ(f.ToString(), "3");
  EXPECT_EQ(f.ToHexString(), "0x3");
}

TEST_F(PrimeFieldTest, Zero) {
  EXPECT_TRUE(GF7::Zero().IsZero());
  EXPECT_FALSE(GF7::One().IsZero());
}

TEST_F(PrimeFieldTest, One) {
  EXPECT_TRUE(GF7::One().IsOne());
  EXPECT_FALSE(GF7::Zero().IsOne());
}

TEST_F(PrimeFieldTest, EqualityOperators) {
  GF7 f(3);
  GF7 f2(4);
  EXPECT_TRUE(f == f);
  EXPECT_TRUE(f != f2);
}

TEST_F(PrimeFieldTest, ComparisonOperator) {
  GF7 f(3);
  GF7 f2(4);
  EXPECT_TRUE(f < f2);
  EXPECT_TRUE(f <= f2);
  EXPECT_FALSE(f > f2);
  EXPECT_FALSE(f >= f2);
}

TEST_F(PrimeFieldTest, AdditiveOperators) {
  struct {
    GF7 a;
    GF7 b;
    GF7 sum;
    GF7 amb;
    GF7 bma;
  } tests[] = {
      {GF7(3), GF7(2), GF7(5), GF7(1), GF7(6)},
      {GF7(5), GF7(3), GF7(1), GF7(2), GF7(5)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);

    GF7 tmp = test.a;
    tmp += test.b;
    EXPECT_EQ(tmp, test.sum);
    tmp -= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(PrimeFieldTest, AdditiveGroupOperators) {
  GF7 f(3);
  EXPECT_EQ(f.Negative(), GF7(4));
  f.NegInPlace();
  EXPECT_EQ(f, GF7(4));

  f = GF7(3);
  EXPECT_EQ(f.Double(), GF7(6));
  f.DoubleInPlace();
  EXPECT_EQ(f, GF7(6));
}

TEST_F(PrimeFieldTest, MultiplicativeOperators) {
  struct {
    GF7 a;
    GF7 b;
    GF7 mul;
    GF7 adb;
    GF7 bda;
  } tests[] = {
      {GF7(3), GF7(2), GF7(6), GF7(5), GF7(3)},
      {GF7(5), GF7(3), GF7(1), GF7(4), GF7(2)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a * test.b, test.mul);
    EXPECT_EQ(test.b * test.a, test.mul);
    EXPECT_EQ(test.a / test.b, test.adb);
    EXPECT_EQ(test.b / test.a, test.bda);

    GF7 tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    tmp /= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(PrimeFieldTest, MultiplicativeGroupOperators) {
  for (int i = 1; i < 7; ++i) {
    GF7 f(i);
    EXPECT_EQ(f * f.Inverse(), GF7::One());
    GF7 f_tmp = f;
    f.InverseInPlace();
    EXPECT_EQ(f * f_tmp, GF7::One());
  }

  GF7 f(3);
  EXPECT_EQ(f.Square(), GF7(2));
  f.SquareInPlace();
  EXPECT_EQ(f, GF7(2));

  f = GF7(3);
  EXPECT_EQ(f.Pow(5), GF7(5));
}

TEST_F(PrimeFieldTest, SumOfProducts) {
  const GF7 a[] = {GF7(3), GF7(2)};
  const GF7 b[] = {GF7(2), GF7(5)};
  EXPECT_EQ(GF7::SumOfProducts(a, b), GF7(2));
}

TEST_F(PrimeFieldTest, Random) {
  bool success = false;
  GF7 r = GF7::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != GF7::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(PrimeFieldTest, DivBy2Exp) {
  struct {
    int v;
  } tests[] = {
      {5},
      {0},
  };

  for (const auto& test : tests) {
    GF7 p(test.v);
    for (size_t i = 0; i < GF7::kModulusBits; ++i) {
      mpz_class q = p.DivBy2Exp(i);
      EXPECT_EQ(q, mpz_class(test.v / (1 << i)));
    }
  }
}

}  // namespace math
}  // namespace tachyon
