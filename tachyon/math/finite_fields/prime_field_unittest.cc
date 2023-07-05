#include "tachyon/math/finite_fields/prime_field.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

namespace {

class PrimeFieldTest : public ::testing::Test {
 public:
  PrimeFieldTest() { Fp7::Init(); }
  PrimeFieldTest(const PrimeFieldTest&) = delete;
  PrimeFieldTest& operator=(const PrimeFieldTest&) = delete;
  ~PrimeFieldTest() override = default;
};

}  // namespace

TEST_F(PrimeFieldTest, Normalize) { EXPECT_EQ(Fp7(-2), Fp7(5)); }

TEST_F(PrimeFieldTest, FromString) {
  EXPECT_EQ(Fp7::FromDecString("3"), Fp7(3));
  EXPECT_EQ(Fp7::FromHexString("0x3"), Fp7(3));
}

TEST_F(PrimeFieldTest, ToString) {
  Fp7 f(3);

  EXPECT_EQ(f.ToString(), "3");
  EXPECT_EQ(f.ToHexString(), "0x3");
}

TEST_F(PrimeFieldTest, Zero) {
  EXPECT_TRUE(Fp7::Zero().IsZero());
  EXPECT_FALSE(Fp7::One().IsZero());
}

TEST_F(PrimeFieldTest, One) {
  EXPECT_TRUE(Fp7::One().IsOne());
  EXPECT_FALSE(Fp7::Zero().IsOne());
}

TEST_F(PrimeFieldTest, ToIntegers) {
  {
    uint64_t out;
    EXPECT_TRUE(Fp7(1).ToUint64(&out));
    EXPECT_EQ(out, 1);
  }
  {
    int64_t out;
    EXPECT_TRUE(Fp7(1).ToInt64(&out));
    EXPECT_EQ(out, 1);
  }
}

TEST_F(PrimeFieldTest, EqualityOperators) {
  Fp7 f(3);
  Fp7 f2(4);
  EXPECT_TRUE(f == f);
  EXPECT_TRUE(f != f2);
}

TEST_F(PrimeFieldTest, ComparisonOperator) {
  Fp7 f(3);
  Fp7 f2(4);
  EXPECT_TRUE(f < f2);
  EXPECT_TRUE(f <= f2);
  EXPECT_FALSE(f > f2);
  EXPECT_FALSE(f >= f2);
}

TEST_F(PrimeFieldTest, AdditiveOperators) {
  struct {
    Fp7 a;
    Fp7 b;
    Fp7 sum;
    Fp7 amb;
    Fp7 bma;
  } tests[] = {
      {Fp7(3), Fp7(2), Fp7(5), Fp7(1), Fp7(6)},
      {Fp7(5), Fp7(3), Fp7(1), Fp7(2), Fp7(5)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);

    Fp7 tmp = test.a;
    tmp += test.b;
    EXPECT_EQ(tmp, test.sum);
    tmp -= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(PrimeFieldTest, AdditiveGroupOperators) {
  Fp7 f(3);
  EXPECT_EQ(f.Negative(), Fp7(-3));
  f.NegativeInPlace();
  EXPECT_EQ(f, Fp7(-3));

  f = Fp7(3);
  EXPECT_EQ(f.Double(), Fp7(6));
  f.DoubleInPlace();
  EXPECT_EQ(f, Fp7(6));
}

TEST_F(PrimeFieldTest, MultiplicativeOperators) {
  struct {
    Fp7 a;
    Fp7 b;
    Fp7 mul;
    Fp7 adb;
    Fp7 bda;
  } tests[] = {
      {Fp7(3), Fp7(2), Fp7(6), Fp7(5), Fp7(3)},
      {Fp7(5), Fp7(3), Fp7(1), Fp7(4), Fp7(2)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a * test.b, test.mul);
    EXPECT_EQ(test.b * test.a, test.mul);
    EXPECT_EQ(test.a / test.b, test.adb);
    EXPECT_EQ(test.b / test.a, test.bda);

    Fp7 tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    tmp /= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(PrimeFieldTest, MultiplicativeGroupOperators) {
  for (int i = 1; i < 7; ++i) {
    Fp7 f(i);
    EXPECT_EQ(f * f.Inverse(), Fp7::One());
    Fp7 f_tmp = f;
    f.InverseInPlace();
    EXPECT_EQ(f * f_tmp, Fp7::One());
  }

  Fp7 f(3);
  EXPECT_EQ(f.Square(), Fp7(2));
  f.SquareInPlace();
  EXPECT_EQ(f, Fp7(2));

  f = Fp7(3);
  EXPECT_EQ(f.Pow(5), Fp7(5));
}

TEST_F(PrimeFieldTest, SumOfProducts) {
  const Fp7 a[] = {Fp7(3), Fp7(2)};
  const Fp7 b[] = {Fp7(2), Fp7(5)};
  EXPECT_EQ(Fp7::SumOfProducts(std::begin(a), std::end(a), std::begin(b),
                               std::end(b)),
            Fp7(2));
}

TEST_F(PrimeFieldTest, Random) {
  bool success = false;
  Fp7 r = Fp7::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != Fp7::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(PrimeFieldTest, ModOperators) {
  Fp7 p(5);
  EXPECT_EQ(p % 3, 2);
}

}  // namespace math
}  // namespace tachyon
