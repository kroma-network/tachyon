#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

namespace {

template <typename PrimeFieldType>
class PrimeFieldTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
#if defined(TACHYON_GMP_BACKEND)
    if constexpr (std::is_same_v<PrimeFieldType, GF7Gmp>) {
      PrimeFieldType::Init();
    }
#endif  // defined(TACHYON_GMP_BACKEND)
  }
};

}  // namespace

#if defined(TACHYON_GMP_BACKEND)
using PrimeFieldTypes = testing::Types<GF7, GF7Gmp>;
#else
using PrimeFieldTypes = testing::Types<GF7>;
#endif
TYPED_TEST_SUITE(PrimeFieldTest, PrimeFieldTypes);

TYPED_TEST(PrimeFieldTest, FromString) {
  using F = TypeParam;

  EXPECT_EQ(F::FromDecString("3"), F(3));
  EXPECT_EQ(F::FromHexString("0x3"), F(3));
}

TYPED_TEST(PrimeFieldTest, ToString) {
  using F = TypeParam;

  F f(3);

  EXPECT_EQ(f.ToString(), "3");
  EXPECT_EQ(f.ToHexString(), "0x3");
}

TYPED_TEST(PrimeFieldTest, Zero) {
  using F = TypeParam;

  EXPECT_TRUE(F::Zero().IsZero());
  EXPECT_FALSE(F::One().IsZero());
}

TYPED_TEST(PrimeFieldTest, One) {
  using F = TypeParam;

  EXPECT_TRUE(F::One().IsOne());
  EXPECT_FALSE(F::Zero().IsOne());
  EXPECT_EQ(F::Config::kOne, F(1).ToMontgomery());
}

TYPED_TEST(PrimeFieldTest, BigIntConversion) {
  using F = TypeParam;

  F r = F::Random();
  EXPECT_EQ(F::FromBigInt(r.ToBigInt()), r);
}

TYPED_TEST(PrimeFieldTest, MontgomeryConversion) {
  using F = TypeParam;

  F r = F::Random();
  EXPECT_EQ(F::FromMontgomery(r.ToMontgomery()), r);
}

TYPED_TEST(PrimeFieldTest, MpzClassConversion) {
  using F = TypeParam;

  F r = F::Random();
  EXPECT_EQ(F::FromMpzClass(r.ToMpzClass()), r);
}

TYPED_TEST(PrimeFieldTest, EqualityOperators) {
  using F = TypeParam;

  F f(3);
  F f2(4);
  EXPECT_TRUE(f == f);
  EXPECT_TRUE(f != f2);
}

TYPED_TEST(PrimeFieldTest, ComparisonOperator) {
  using F = TypeParam;

  F f(3);
  F f2(4);
  EXPECT_TRUE(f < f2);
  EXPECT_TRUE(f <= f2);
  EXPECT_FALSE(f > f2);
  EXPECT_FALSE(f >= f2);
}

TYPED_TEST(PrimeFieldTest, AdditiveOperators) {
  using F = TypeParam;

  struct {
    F a;
    F b;
    F sum;
    F amb;
    F bma;
  } tests[] = {
      {F(3), F(2), F(5), F(1), F(6)},
      {F(5), F(3), F(1), F(2), F(5)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a + test.b, test.sum);
    EXPECT_EQ(test.b + test.a, test.sum);
    EXPECT_EQ(test.a - test.b, test.amb);
    EXPECT_EQ(test.b - test.a, test.bma);

    F tmp = test.a;
    tmp += test.b;
    EXPECT_EQ(tmp, test.sum);
    tmp -= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TYPED_TEST(PrimeFieldTest, AdditiveGroupOperators) {
  using F = TypeParam;

  F f(3);
  EXPECT_EQ(f.Negative(), F(4));
  f.NegInPlace();
  EXPECT_EQ(f, F(4));

  f = F(3);
  EXPECT_EQ(f.Double(), F(6));
  f.DoubleInPlace();
  EXPECT_EQ(f, F(6));
}

TYPED_TEST(PrimeFieldTest, MultiplicativeOperators) {
  using F = TypeParam;

  struct {
    F a;
    F b;
    F mul;
    F adb;
    F bda;
  } tests[] = {
      {F(3), F(2), F(6), F(5), F(3)},
      {F(5), F(3), F(1), F(4), F(2)},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.a * test.b, test.mul);
    EXPECT_EQ(test.b * test.a, test.mul);
    EXPECT_EQ(test.a / test.b, test.adb);
    EXPECT_EQ(test.b / test.a, test.bda);

    F tmp = test.a;
    tmp *= test.b;
    EXPECT_EQ(tmp, test.mul);
    tmp /= test.b;
    EXPECT_EQ(tmp, test.a);
  }
}

TYPED_TEST(PrimeFieldTest, MultiplicativeGroupOperators) {
  using F = TypeParam;

  for (int i = 1; i < 7; ++i) {
    F f(i);
    EXPECT_EQ(f * f.Inverse(), F::One());
    F f_tmp = f;
    f.InverseInPlace();
    EXPECT_EQ(f * f_tmp, F::One());
  }

  F f(3);
  EXPECT_EQ(f.Square(), F(2));
  f.SquareInPlace();
  EXPECT_EQ(f, F(2));

  f = F(3);
  EXPECT_EQ(f.Pow(5), F(5));
}

TYPED_TEST(PrimeFieldTest, SumOfProducts) {
  using F = TypeParam;

  const F a[] = {F(3), F(2)};
  const F b[] = {F(2), F(5)};
  EXPECT_EQ(F::SumOfProducts(a, b), F(2));
}

TYPED_TEST(PrimeFieldTest, Random) {
  using F = TypeParam;

  bool success = false;
  F r = F::Random();
  for (size_t i = 0; i < 100; ++i) {
    if (r != F::Random()) {
      success = true;
      break;
    }
  }
  EXPECT_TRUE(success);
}

TYPED_TEST(PrimeFieldTest, DivBy2Exp) {
  using F = TypeParam;

  struct {
    int v;
  } tests[] = {
      {5},
      {0},
  };

  for (const auto& test : tests) {
    F p(test.v);
    for (size_t i = 0; i < F::kModulusBits; ++i) {
      BigInt<1> q = p.DivBy2Exp(i);
      EXPECT_EQ(q, BigInt<1>(test.v / (1 << i)));
    }
  }
}

TYPED_TEST(PrimeFieldTest, Copyable) {
  using F = TypeParam;

  const F expected = F::Random();
  F value;

  base::VectorBuffer write_buf;
  EXPECT_TRUE(write_buf.Write(expected));

  write_buf.set_buffer_offset(0);
  write_buf.Read(&value);
  EXPECT_EQ(expected, value);
}

}  // namespace tachyon::math
