#include <optional>

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7_3.h"

namespace tachyon::math {

namespace {

class CubicExtensionFieldTest : public FiniteFieldTest<GF7_3> {};

}  // namespace

TEST_F(CubicExtensionFieldTest, Zero) {
  EXPECT_TRUE(GF7_3::Zero().IsZero());
  EXPECT_FALSE(GF7_3::One().IsZero());
}

TEST_F(CubicExtensionFieldTest, One) {
  EXPECT_TRUE(GF7_3::One().IsOne());
  EXPECT_FALSE(GF7_3::Zero().IsOne());
}

TEST_F(CubicExtensionFieldTest, Random) {
  bool success = false;
  GF7_3 r = GF7_3::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != GF7_3::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(CubicExtensionFieldTest, Norm) {
  constexpr static uint32_t kModulus = GF7::Config::kModulus;
  GF7_3 r = GF7_3::Random();
  GF7_3 r_to_p = r.Pow(kModulus);
  GF7_3 r_to_p2 = r_to_p.Pow(kModulus);
  EXPECT_EQ(r.Norm(), (r * r_to_p * r_to_p2).c0());
}

TEST_F(CubicExtensionFieldTest, EqualityOperators) {
  GF7_3 f(GF7(3), GF7(4), GF7(5));
  GF7_3 f2(GF7(4), GF7(4), GF7(5));
  EXPECT_FALSE(f == f2);
  EXPECT_TRUE(f != f2);

  GF7_3 f3(GF7(4), GF7(3), GF7(5));
  EXPECT_FALSE(f2 == f3);
  EXPECT_TRUE(f2 != f3);

  GF7_3 f4(GF7(3), GF7(4), GF7(6));
  EXPECT_FALSE(f == f4);
  EXPECT_TRUE(f != f4);

  GF7_3 f5(GF7(3), GF7(4), GF7(5));
  EXPECT_TRUE(f == f5);
}

TEST_F(CubicExtensionFieldTest, ComparisonOperator) {
  GF7_3 f(GF7(3), GF7(4), GF7(5));
  GF7_3 f2(GF7(4), GF7(4), GF7(5));
  EXPECT_LT(f, f2);
  EXPECT_LE(f, f2);
  EXPECT_GT(f2, f);
  EXPECT_GE(f2, f);

  GF7_3 f3(GF7(4), GF7(3), GF7(5));
  GF7_3 f4(GF7(3), GF7(4), GF7(5));
  EXPECT_LT(f3, f4);
  EXPECT_LE(f3, f4);
  EXPECT_GT(f4, f3);
  EXPECT_GE(f4, f3);

  GF7_3 f5(GF7(4), GF7(5), GF7(3));
  GF7_3 f6(GF7(3), GF7(2), GF7(5));
  EXPECT_LT(f5, f6);
  EXPECT_LE(f5, f6);
  EXPECT_GT(f6, f5);
  EXPECT_GE(f6, f5);
}

TEST_F(CubicExtensionFieldTest, AdditiveOperators) {
  struct {
    GF7_3 a;
    GF7_3 b;
    GF7_3 sum;
    GF7_3 amb;
    GF7_3 bma;
  } tests[] = {
      {
          {GF7(1), GF7(2), GF7(3)},
          {GF7(3), GF7(5), GF7(6)},
          {GF7(4), GF7(0), GF7(2)},
          {GF7(5), GF7(4), GF7(4)},
          {GF7(2), GF7(3), GF7(3)},
      },
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);

    GF7_3 tmp = test.a;
    tmp += test.b;
    EXPECT_EQ(tmp, test.sum);
    tmp -= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(CubicExtensionFieldTest, AdditiveGroupOperators) {
  GF7_3 f(GF7(3), GF7(4), GF7(5));
  GF7_3 f_neg(GF7(4), GF7(3), GF7(2));
  EXPECT_EQ(-f, f_neg);
  f.NegateInPlace();
  EXPECT_EQ(f, f_neg);

  f = GF7_3(GF7(3), GF7(4), GF7(5));
  GF7_3 f_dbl(GF7(6), GF7(1), GF7(3));
  EXPECT_EQ(f.Double(), f_dbl);
  f.DoubleInPlace();
  EXPECT_EQ(f, f_dbl);
}

TEST_F(CubicExtensionFieldTest, MultiplicativeOperators) {
  struct {
    GF7_3 a;
    GF7_3 b;
    GF7_3 mul;
    GF7_3 adb;
    GF7_3 bda;
  } tests[] = {
      {
          {GF7(1), GF7(2), GF7(3)},
          {GF7(3), GF7(5), GF7(6)},
          {GF7(1), GF7(5), GF7(4)},
          {GF7(3), GF7(3), GF7(4)},
          {GF7(3), GF7(1), GF7(2)},
      },
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a * test.b, test.mul);
    EXPECT_EQ(test.b * test.a, test.mul);
    EXPECT_EQ(test.a / test.b, test.adb);
    EXPECT_EQ(test.b / test.a, test.bda);

    GF7_3 tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    ASSERT_TRUE(tmp /= test.b);
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(CubicExtensionFieldTest, MultiplicativeOperators2) {
  GF7_3 f(GF7(3), GF7(4), GF7(5));
  GF7_3 f_mul(GF7(6), GF7(1), GF7(3));
  EXPECT_EQ(f * GF7(2), f_mul);
  f *= GF7(2);
  EXPECT_EQ(f, f_mul);
}

TEST_F(CubicExtensionFieldTest, MultiplicativeGroupOperators) {
  GF7_3 f = GF7_3::Random();
  std::optional<GF7_3> f_inv = f.Inverse();
  if (UNLIKELY(f.IsZero())) {
    ASSERT_FALSE(f_inv);
    ASSERT_FALSE(f.InverseInPlace());
  } else {
    EXPECT_EQ(f * *f_inv, GF7_3::One());
    GF7_3 f_tmp = f;
    EXPECT_EQ(**f.InverseInPlace() * f_tmp, GF7_3::One());
  }

  f = GF7_3(GF7(3), GF7(4), GF7(5));
  GF7_3 f_sqr = GF7_3(GF7(5), GF7(4), GF7(4));
  EXPECT_EQ(f.Square(), f_sqr);
  f.SquareInPlace();
  EXPECT_EQ(f, f_sqr);
}

TEST_F(CubicExtensionFieldTest, JsonValueConverter) {
  GF7_3 expected_point(GF7(1), GF7(2), GF7(3));
  std::string expected_json = R"({"c0":1,"c1":2,"c2":3})";

  GF7_3 p;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &p, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(p, expected_point);

  std::string json = base::WriteToJson(p);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
