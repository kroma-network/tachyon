#include "gtest/gtest.h"

#include "tachyon/base/buffer/vector_buffer.h"
#include "tachyon/base/optional.h"
// #include "tachyon/math/finite_fields/prime_field_gpu_debug.h"
#include "tachyon/base/auto_reset.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"
#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

namespace {

template <typename PrimeField>
class PrimeFieldTest : public FiniteFieldTest<PrimeField> {};

}  // namespace

// TODO(chokobole): Enable test for |PrimeFieldGpuDebug<GF7Config>|.
// Since |PrimeFieldGpuDebug| only supports the montgomery version and |GF7|
// disables montgomery, this makes the test fail.
using PrimeFieldTypes = testing::Types<GF7 /*, PrimeFieldGpuDebug<GF7Config>*/>;
TYPED_TEST_SUITE(PrimeFieldTest, PrimeFieldTypes);

TYPED_TEST(PrimeFieldTest, FromString) {
  using F = TypeParam;

  EXPECT_EQ(*F::FromDecString("3"), F(3));
  EXPECT_FALSE(F::FromDecString("a").has_value());
  EXPECT_EQ(*F::FromHexString("0x3"), F(3));
  EXPECT_FALSE(F::FromHexString("a").has_value());
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
  EXPECT_EQ(F::Config::kOne, F(1).value());
}

TYPED_TEST(PrimeFieldTest, BigIntConversion) {
  using F = TypeParam;

  F r = F::Random();
  EXPECT_EQ(F::FromBigInt(r.ToBigInt()), r);
}

TYPED_TEST(PrimeFieldTest, EqualityOperators) {
  using F = TypeParam;

  F f(3);
  F f2(4);
  EXPECT_EQ(f, f);
  EXPECT_NE(f, f2);
}

TYPED_TEST(PrimeFieldTest, ComparisonOperator) {
  using F = TypeParam;

  F f(3);
  F f2(4);
  EXPECT_LT(f, f2);
  EXPECT_LE(f, f2);
  EXPECT_GT(f2, f);
  EXPECT_GE(f2, f);
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
  EXPECT_EQ(-f, F(4));
  f.NegateInPlace();
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
    ASSERT_TRUE(tmp /= test.b);
    EXPECT_EQ(tmp, test.a);
  }
}

TYPED_TEST(PrimeFieldTest, MultiplicativeGroupOperators) {
  using F = TypeParam;

  for (int i = 0; i < 7; ++i) {
    F f(i);
    std::optional<F> f_inv = f.Inverse();
    if (f.IsZero()) {
      ASSERT_FALSE(f_inv);
      ASSERT_FALSE(f.InverseInPlace());
    } else {
      EXPECT_EQ(f * *f_inv, F::One());
      F f_tmp = f;
      EXPECT_EQ(**f.InverseInPlace() * f_tmp, F::One());
    }
  }

  F f(3);
  EXPECT_EQ(f.Square(), F(2));
  f.SquareInPlace();
  EXPECT_EQ(f, F(2));

  f = F(3);
  EXPECT_EQ(f.Pow(5), F(5));
}

TYPED_TEST(PrimeFieldTest, SumOfProductsSerial) {
  using F = TypeParam;

  const F a[] = {F(3), F(2)};
  const F b[] = {F(2), F(5)};
  EXPECT_EQ(F::SumOfProductsSerial(a, b), F(2));
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

TYPED_TEST(PrimeFieldTest, Copyable) {
  using F = TypeParam;

  const F expected = F::Random();

  for (size_t i = 0; i < 2; ++i) {
    bool s_is_in_montgomery = i == 0;
    SCOPED_TRACE(
        absl::Substitute("s_is_in_montgomery: $0", s_is_in_montgomery));
    base::AutoReset<bool> auto_reset(&base::Copyable<F>::s_is_in_montgomery,
                                     s_is_in_montgomery);
    base::Uint8VectorBuffer write_buf;
    ASSERT_TRUE(write_buf.Grow(base::EstimateSize(expected)));
    ASSERT_TRUE(write_buf.Write(expected));
    ASSERT_TRUE(write_buf.Done());

    write_buf.set_buffer_offset(0);

    F value;
    ASSERT_TRUE(write_buf.Read(&value));
    EXPECT_EQ(expected, value);
  }
}

}  // namespace tachyon::math
