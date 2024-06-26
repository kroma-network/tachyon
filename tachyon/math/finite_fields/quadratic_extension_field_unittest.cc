#include <optional>

#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fq6.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7_2.h"

namespace tachyon::math {

namespace {

class QuadraticExtensionFieldTest : public FiniteFieldTest<GF7_2> {};

}  // namespace

TEST_F(QuadraticExtensionFieldTest, Zero) {
  EXPECT_TRUE(GF7_2::Zero().IsZero());
  EXPECT_FALSE(GF7_2::One().IsZero());
}

TEST_F(QuadraticExtensionFieldTest, One) {
  EXPECT_TRUE(GF7_2::One().IsOne());
  EXPECT_FALSE(GF7_2::Zero().IsOne());
}

TEST_F(QuadraticExtensionFieldTest, Random) {
  bool success = false;
  GF7_2 r = GF7_2::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != GF7_2::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(QuadraticExtensionFieldTest, Norm) {
  constexpr static uint32_t kModulus = GF7::Config::kModulus;
  GF7_2 r = GF7_2::Random();
  GF7_2 r_to_p = r.Pow(kModulus);
  EXPECT_EQ(r.Norm(), (r * r_to_p).c0());
}

TEST_F(QuadraticExtensionFieldTest, ConjugateInPlace) {
  GF7_2 f = GF7_2::Random();
  GF7_2 f2 = f;
  f2.ConjugateInPlace();
  EXPECT_EQ(f.c0(), f2.c0());
  EXPECT_EQ(f.c1(), -f2.c1());
}

TEST_F(QuadraticExtensionFieldTest, EqualityOperators) {
  GF7_2 f(GF7(3), GF7(4));
  GF7_2 f2(GF7(4), GF7(4));
  EXPECT_NE(f, f2);

  GF7_2 f3(GF7(4), GF7(3));
  EXPECT_NE(f2, f3);

  GF7_2 f4(GF7(4), GF7(4));
  EXPECT_EQ(f2, f4);
}

TEST_F(QuadraticExtensionFieldTest, ComparisonOperator) {
  GF7_2 f(GF7(3), GF7(4));
  GF7_2 f2(GF7(4), GF7(4));
  EXPECT_LT(f, f2);
  EXPECT_LE(f, f2);
  EXPECT_GT(f2, f);
  EXPECT_GE(f2, f);

  GF7_2 f3(GF7(4), GF7(3));
  GF7_2 f4(GF7(4), GF7(4));
  EXPECT_LT(f3, f4);
  EXPECT_LE(f3, f4);
  EXPECT_GT(f4, f3);
  EXPECT_GE(f4, f3);
}

TEST_F(QuadraticExtensionFieldTest, AdditiveOperators) {
  struct {
    GF7_2 a;
    GF7_2 b;
    GF7_2 sum;
    GF7_2 amb;
    GF7_2 bma;
  } tests[] = {
      {
          {GF7(1), GF7(2)},
          {GF7(3), GF7(5)},
          {GF7(4), GF7(0)},
          {GF7(5), GF7(4)},
          {GF7(2), GF7(3)},
      },
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);

    GF7_2 tmp = test.a;
    tmp += test.b;
    EXPECT_EQ(tmp, test.sum);
    tmp -= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(QuadraticExtensionFieldTest, AdditiveGroupOperators) {
  GF7_2 f(GF7(3), GF7(4));
  GF7_2 f_neg(GF7(4), GF7(3));
  EXPECT_EQ(-f, f_neg);
  f.NegateInPlace();
  EXPECT_EQ(f, f_neg);

  f = GF7_2(GF7(3), GF7(4));
  GF7_2 f_dbl(GF7(6), GF7(1));
  EXPECT_EQ(f.Double(), f_dbl);
  f.DoubleInPlace();
  EXPECT_EQ(f, f_dbl);
}

TEST_F(QuadraticExtensionFieldTest, MultiplicativeOperators) {
  struct {
    GF7_2 a;
    GF7_2 b;
    GF7_2 mul;
    GF7_2 adb;
    GF7_2 bda;
  } tests[] = {
      {
          {GF7(1), GF7(2)},
          {GF7(3), GF7(5)},
          {GF7(0), GF7(4)},
          {GF7(1), GF7(6)},
          {GF7(4), GF7(4)},
      },
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a * test.b, test.mul);
    EXPECT_EQ(test.b * test.a, test.mul);
    EXPECT_EQ(test.a / test.b, test.adb);
    EXPECT_EQ(test.b / test.a, test.bda);

    GF7_2 tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    ASSERT_TRUE(tmp /= test.b);
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(QuadraticExtensionFieldTest, MultiplicativeOperators2) {
  GF7_2 f(GF7(3), GF7(4));
  GF7_2 f_mul(GF7(6), GF7(1));
  EXPECT_EQ(f * GF7(2), f_mul);
  f *= GF7(2);
  EXPECT_EQ(f, f_mul);
}

TEST_F(QuadraticExtensionFieldTest, MultiplicativeGroupOperators) {
  GF7_2 f = GF7_2::Random();
  std::optional<GF7_2> f_inv = f.Inverse();
  if (UNLIKELY(f.IsZero())) {
    ASSERT_FALSE(f_inv);
    ASSERT_FALSE(f.InverseInPlace());
  } else {
    EXPECT_EQ(f * *f_inv, GF7_2::One());
    GF7_2 f_tmp = f;
    EXPECT_EQ(**f.InverseInPlace() * f_tmp, GF7_2::One());
  }

  f = GF7_2(GF7(3), GF7(4));
  GF7_2 f_sqr = GF7_2(GF7(0), GF7(3));
  EXPECT_EQ(f.Square(), f_sqr);
  f.SquareInPlace();
  EXPECT_EQ(f, f_sqr);
}

TEST(CyclotomicInverseTest, FastCyclotomicInverse) {
  bn254::Fq6::Init();
  bn254::Fq6 f = bn254::Fq6::Random();
  bn254::Fq6 f_tmp = f;
  if (UNLIKELY(f.IsZero())) {
    ASSERT_FALSE(f.InverseInPlace());
    ASSERT_FALSE(f_tmp.CyclotomicInverseInPlace());
  } else {
    ASSERT_TRUE(f.InverseInPlace());
    ASSERT_TRUE(f_tmp.CyclotomicInverseInPlace());
  }
  EXPECT_EQ(f, f_tmp);
}

TEST_F(QuadraticExtensionFieldTest, JsonValueConverter) {
  GF7_2 expected_point(GF7(1), GF7(2));
  std::string expected_json = R"({"c0":1,"c1":2})";

  GF7_2 p;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &p, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(p, expected_point);

  std::string json = base::WriteToJson(p);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
