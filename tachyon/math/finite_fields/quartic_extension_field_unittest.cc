#include <optional>

#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/baby_bear/baby_bear4.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::math {

namespace {

using F4 = BabyBear4;
using F = BabyBear;

class QuaticExtensionFieldTest : public FiniteFieldTest<F4> {};

}  // namespace

TEST_F(QuaticExtensionFieldTest, Zero) {
  EXPECT_TRUE(F4::Zero().IsZero());
  EXPECT_FALSE(F4::One().IsZero());
}

TEST_F(QuaticExtensionFieldTest, One) {
  EXPECT_TRUE(F4::One().IsOne());
  EXPECT_FALSE(F4::Zero().IsOne());
}

TEST_F(QuaticExtensionFieldTest, Random) {
  bool success = false;
  F4 r = F4::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != F4::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TEST_F(QuaticExtensionFieldTest, Norm) {
  constexpr static uint32_t kModulus = BabyBear::Config::kModulus;
  F4 r = F4::Random();
  F4 r_to_p = r.Pow(kModulus);
  F4 r_to_p2 = r_to_p.Pow(kModulus);
  F4 r_to_p3 = r_to_p2.Pow(kModulus);
  EXPECT_EQ(r.Norm(), (r * r_to_p * r_to_p2 * r_to_p3).c0());
}

TEST_F(QuaticExtensionFieldTest, EqualityOperators) {
  F4 f(F(3), F(4), F(5), F(6));
  F4 f2(F(4), F(4), F(5), F(6));
  EXPECT_NE(f, f2);
  F4 f3(F(4), F(3), F(5), F(6));
  EXPECT_NE(f2, f3);

  F4 f4(F(3), F(4), F(5), F(7));
  EXPECT_NE(f, f4);

  F4 f5(F(3), F(4), F(5), F(6));
  EXPECT_EQ(f, f5);
}

TEST_F(QuaticExtensionFieldTest, ComparisonOperator) {
  F4 f(F(3), F(4), F(5), F(6));
  F4 f2(F(4), F(4), F(5), F(6));
  EXPECT_LT(f, f2);
  EXPECT_LE(f, f2);
  EXPECT_GT(f2, f);
  EXPECT_GE(f2, f);

  F4 f3(F(4), F(3), F(5), F(6));
  F4 f4(F(3), F(4), F(5), F(6));
  EXPECT_LT(f3, f4);
  EXPECT_LE(f3, f4);
  EXPECT_GT(f4, f3);
  EXPECT_GE(f4, f3);

  F4 f5(F(4), F(5), F(6), F(3));
  F4 f6(F(3), F(2), F(6), F(5));
  EXPECT_LT(f5, f6);
  EXPECT_LE(f5, f6);
  EXPECT_GT(f6, f5);
  EXPECT_GE(f6, f5);
}

TEST_F(QuaticExtensionFieldTest, AdditiveOperators) {
  struct {
    F4 a;
    F4 b;
    F4 sum;
    F4 amb;
    F4 bma;
  } tests[] = {
      {
          {F(1), F(2), F(3), F(4)},
          {F(3), F(5), F(6), F(8)},
          {F(4), F(7), F(9), F(12)},
          {-F(2), -F(3), -F(3), -F(4)},
          {F(2), F(3), F(3), F(4)},
      },
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);

    F4 tmp = test.a;
    tmp += test.b;
    EXPECT_EQ(tmp, test.sum);
    tmp -= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(QuaticExtensionFieldTest, AdditiveGroupOperators) {
  F4 f(F(3), F(4), F(5), F(6));
  F4 f_neg(-F(3), -F(4), -F(5), -F(6));
  EXPECT_EQ(-f, f_neg);
  f.NegateInPlace();
  EXPECT_EQ(f, f_neg);

  f = F4(F(3), F(4), F(5), F(6));
  F4 f_dbl(F(6), F(8), F(10), F(12));
  EXPECT_EQ(f.Double(), f_dbl);
  f.DoubleInPlace();
  EXPECT_EQ(f, f_dbl);
}

TEST_F(QuaticExtensionFieldTest, MultiplicativeOperators) {
  struct {
    F4 a;
    F4 b;
    F4 mul;
    F4 adb;
    F4 bda;
  } tests[] = {
      {
          {F(1), F(2), F(3), F(4)},
          {F(3), F(5), F(6), F(8)},
          {F(597), F(539), F(377), F(47)},
          {F(1144494179), F(1502926259), F(1509084158), F(151175067)},
          {F(653096429), F(494869942), F(67683040), F(1807436149)},
      },
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a * test.b, test.mul);
    EXPECT_EQ(test.b * test.a, test.mul);
    EXPECT_EQ(test.a / test.b, test.adb);
    EXPECT_EQ(test.b / test.a, test.bda);

    F4 tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    ASSERT_TRUE(tmp /= test.b);
    EXPECT_EQ(tmp, test.a);
  }
}

TEST_F(QuaticExtensionFieldTest, MultiplicativeOperators2) {
  F4 f(F(3), F(4), F(5), F(6));
  F4 f_mul(F(6), F(8), F(10), F(12));
  EXPECT_EQ(f * F(2), f_mul);
  f *= F(2);
  EXPECT_EQ(f, f_mul);
}

TEST_F(QuaticExtensionFieldTest, MultiplicativeGroupOperators) {
  F4 f = F4::Random();
  std::optional<F4> f_inv = f.Inverse();
  if (UNLIKELY(f.IsZero())) {
    ASSERT_FALSE(f_inv);
    ASSERT_FALSE(f.InverseInPlace());
  } else {
    EXPECT_EQ(f * *f_inv, F4::One());
    F4 f_tmp = f;
    EXPECT_EQ(**f.InverseInPlace() * f_tmp, F4::One());
  }

  f = F4(F(3), F(4), F(5), F(6));
  F4 f_sqr = F4(F(812), F(684), F(442), F(76));
  EXPECT_EQ(f.Square(), f_sqr);
  f.SquareInPlace();
  EXPECT_EQ(f, f_sqr);
}

TEST_F(QuaticExtensionFieldTest, JsonValueConverter) {
  F4 expected_point(F(1), F(2), F(3), F(4));
  std::string expected_json = R"({"c0":1,"c1":2,"c2":3,"c3":4})";

  F4 p;
  std::string error;
  ASSERT_TRUE(base::ParseJson(expected_json, &p, &error));
  ASSERT_TRUE(error.empty());
  EXPECT_EQ(p, expected_point);

  std::string json = base::WriteToJson(p);
  EXPECT_EQ(json, expected_json);
}

}  // namespace tachyon::math
