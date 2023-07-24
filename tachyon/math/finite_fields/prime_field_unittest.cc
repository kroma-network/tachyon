#include "tachyon/math/finite_fields/prime_field.h"

#include "gtest/gtest.h"

namespace tachyon {
namespace math {

namespace {

template <typename PrimeFieldType>
class PrimeFieldTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    if constexpr (std::is_same_v<PrimeFieldType, GF7Gmp>) {
      PrimeFieldType::Init();
    }
  }
};

}  // namespace

#if defined(TACHYON_GMP_BACKEND)
using PrimeFiledTypes = testing::Types<GF7, GF7Gmp>;
#else
using PrimeFiledTypes = testing::Types<GF7>;
#endif
TYPED_TEST_SUITE(PrimeFieldTest, PrimeFiledTypes);

TYPED_TEST(PrimeFieldTest, FromString) {
  EXPECT_EQ(TypeParam::FromDecString("3"), TypeParam(3));
  EXPECT_EQ(TypeParam::FromHexString("0x3"), TypeParam(3));
}

TYPED_TEST(PrimeFieldTest, ToString) {
  TypeParam f(3);

  EXPECT_EQ(f.ToString(), "3");
  EXPECT_EQ(f.ToHexString(), "0x3");
}

TYPED_TEST(PrimeFieldTest, Zero) {
  EXPECT_TRUE(TypeParam::Zero().IsZero());
  EXPECT_FALSE(TypeParam::One().IsZero());
}

TYPED_TEST(PrimeFieldTest, One) {
  EXPECT_TRUE(TypeParam::One().IsOne());
  EXPECT_FALSE(TypeParam::Zero().IsOne());
  EXPECT_EQ(TypeParam::Config::kOne, TypeParam(1).ToMontgomery());
}

TYPED_TEST(PrimeFieldTest, BigIntConversion) {
  TypeParam r = TypeParam::Random();
  EXPECT_EQ(TypeParam::FromBigInt(r.ToBigInt()), r);
}

TYPED_TEST(PrimeFieldTest, MontgomeryConversion) {
  TypeParam r = TypeParam::Random();
  EXPECT_EQ(TypeParam::FromMontgomery(r.ToMontgomery()), r);
}

TYPED_TEST(PrimeFieldTest, EqualityOperators) {
  TypeParam f(3);
  TypeParam f2(4);
  EXPECT_TRUE(f == f);
  EXPECT_TRUE(f != f2);
}

TYPED_TEST(PrimeFieldTest, ComparisonOperator) {
  TypeParam f(3);
  TypeParam f2(4);
  EXPECT_TRUE(f < f2);
  EXPECT_TRUE(f <= f2);
  EXPECT_FALSE(f > f2);
  EXPECT_FALSE(f >= f2);
}

TYPED_TEST(PrimeFieldTest, AdditiveOperators) {
  struct {
    TypeParam a;
    TypeParam b;
    TypeParam sum;
    TypeParam amb;
    TypeParam bma;
  } tests[] = {
      {TypeParam(3), TypeParam(2), TypeParam(5), TypeParam(1), TypeParam(6)},
      {TypeParam(5), TypeParam(3), TypeParam(1), TypeParam(2), TypeParam(5)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);

    TypeParam tmp = test.a;
    tmp += test.b;
    EXPECT_EQ(tmp, test.sum);
    tmp -= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TYPED_TEST(PrimeFieldTest, AdditiveGroupOperators) {
  TypeParam f(3);
  EXPECT_EQ(f.Negative(), TypeParam(4));
  f.NegInPlace();
  EXPECT_EQ(f, TypeParam(4));

  f = TypeParam(3);
  EXPECT_EQ(f.Double(), TypeParam(6));
  f.DoubleInPlace();
  EXPECT_EQ(f, TypeParam(6));
}

TYPED_TEST(PrimeFieldTest, MultiplicativeOperators) {
  struct {
    TypeParam a;
    TypeParam b;
    TypeParam mul;
    TypeParam adb;
    TypeParam bda;
  } tests[] = {
      {TypeParam(3), TypeParam(2), TypeParam(6), TypeParam(5), TypeParam(3)},
      {TypeParam(5), TypeParam(3), TypeParam(1), TypeParam(4), TypeParam(2)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a * test.b, test.mul);
    EXPECT_EQ(test.b * test.a, test.mul);
    EXPECT_EQ(test.a / test.b, test.adb);
    EXPECT_EQ(test.b / test.a, test.bda);

    TypeParam tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    tmp /= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TYPED_TEST(PrimeFieldTest, MultiplicativeGroupOperators) {
  for (int i = 1; i < 7; ++i) {
    TypeParam f(i);
    EXPECT_EQ(f * f.Inverse(), TypeParam::One());
    TypeParam f_tmp = f;
    f.InverseInPlace();
    EXPECT_EQ(f * f_tmp, TypeParam::One());
  }

  TypeParam f(3);
  EXPECT_EQ(f.Square(), TypeParam(2));
  f.SquareInPlace();
  EXPECT_EQ(f, TypeParam(2));

  f = TypeParam(3);
  EXPECT_EQ(f.Pow(TypeParam(5).ToBigInt()), TypeParam(5));
}

TYPED_TEST(PrimeFieldTest, SumOfProducts) {
  const TypeParam a[] = {TypeParam(3), TypeParam(2)};
  const TypeParam b[] = {TypeParam(2), TypeParam(5)};
  EXPECT_EQ(TypeParam::SumOfProducts(a, b), TypeParam(2));
}

TYPED_TEST(PrimeFieldTest, Random) {
  bool success = false;
  TypeParam r = TypeParam::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != TypeParam::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TYPED_TEST(PrimeFieldTest, DivBy2Exp) {
  struct {
    int v;
  } tests[] = {
      {5},
      {0},
  };

  for (const auto& test : tests) {
    TypeParam p(test.v);
    for (size_t i = 0; i < TypeParam::kModulusBits; ++i) {
      mpz_class q = p.DivBy2Exp(i);
      EXPECT_EQ(q, mpz_class(test.v / (1 << i)));
    }
  }
}

}  // namespace math
}  // namespace tachyon
