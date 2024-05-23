#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/binary_fields/binary_fields.h"

namespace tachyon::math {

namespace {
template <typename BinaryField>
class BinaryFieldTest : public testing::Test {
 public:
  static void SetUpTestSuite() { BinaryField::Init(); }
};

}  // namespace

using BinaryFieldTypes =
    testing::Types<BinaryField1, BinaryField2, BinaryField4, BinaryField8>;

TYPED_TEST_SUITE(BinaryFieldTest, BinaryFieldTypes);

TYPED_TEST(BinaryFieldTest, FromString) {
  using BinaryField = TypeParam;
  if constexpr (BinaryField::Config::kModulusBits > 2) {
    EXPECT_EQ(*BinaryField::FromDecString("3"), BinaryField(3));
    EXPECT_FALSE(BinaryField::FromDecString("x").has_value());
    EXPECT_EQ(*BinaryField::FromHexString("0x3"), BinaryField(3));
    EXPECT_FALSE(BinaryField::FromHexString("x").has_value());
  } else {
    GTEST_SKIP() << "Modulus is too small";
  }
}

TYPED_TEST(BinaryFieldTest, ToString) {
  using BinaryField = TypeParam;
  if constexpr (BinaryField::Config::kModulusBits > 2) {
    BinaryField f(3);

    EXPECT_EQ(f.ToString(), "3");
    EXPECT_EQ(f.ToHexString(), "0x3");
  } else {
    GTEST_SKIP() << "Modulus is too small";
  }
}

TYPED_TEST(BinaryFieldTest, Zero) {
  using BinaryField = TypeParam;
  EXPECT_TRUE(BinaryField::Zero().IsZero());
  EXPECT_FALSE(BinaryField::One().IsZero());
}

TYPED_TEST(BinaryFieldTest, One) {
  using BinaryField = TypeParam;
  EXPECT_TRUE(BinaryField::One().IsOne());
  EXPECT_FALSE(BinaryField::Zero().IsOne());
  EXPECT_EQ(BinaryField::Config::kOne, BinaryField(1).value());
}

TYPED_TEST(BinaryFieldTest, BigIntConversion) {
  using BinaryField = TypeParam;
  BinaryField r = BinaryField::Random();
  EXPECT_EQ(BinaryField::FromBigInt(r.ToBigInt()), r);
}

TYPED_TEST(BinaryFieldTest, EqualityOperators) {
  using BinaryField = TypeParam;
  if constexpr (BinaryField::Config::kModulusBits > 3) {
    BinaryField f(3);
    BinaryField f2(4);
    EXPECT_TRUE(f == f);
    EXPECT_TRUE(f != f2);
  } else {
    GTEST_SKIP() << "Modulus is too small";
  }
}

TYPED_TEST(BinaryFieldTest, ComparisonOperator) {
  using BinaryField = TypeParam;
  if constexpr (BinaryField::Config::kModulusBits > 3) {
    BinaryField f(3);
    BinaryField f2(4);
    EXPECT_TRUE(f < f2);
    EXPECT_TRUE(f <= f2);
    EXPECT_FALSE(f > f2);
    EXPECT_FALSE(f >= f2);
  } else {
    GTEST_SKIP() << "Modulus is too small";
  }
}

TYPED_TEST(BinaryFieldTest, AdditiveGroupOperators) {
  using BinaryField = TypeParam;
  BinaryField f = BinaryField::Random();
  SCOPED_TRACE(absl::Substitute("f: $0", f.ToString()));
  BinaryField f_neg = -f;
  EXPECT_TRUE((f_neg + f).IsZero());
  f.NegateInPlace();
  EXPECT_EQ(f, f_neg);

  BinaryField f_double = f.Double();
  EXPECT_EQ(f + f, f_double);
  f.DoubleInPlace();
  EXPECT_EQ(f, f_double);
}

TYPED_TEST(BinaryFieldTest, MultiplicativeGroupOperators) {
  using BinaryField = TypeParam;
  BinaryField f = BinaryField::Random();
  SCOPED_TRACE(absl::Substitute("f: $0", f.ToString()));
  std::optional<BinaryField> f_inv = f.Inverse();
  if (UNLIKELY(f.IsZero())) {
    ASSERT_FALSE(f_inv);
    ASSERT_FALSE(f.InverseInPlace());
  } else {
    EXPECT_EQ(f * *f_inv, BinaryField::One());
    EXPECT_EQ(**f.InverseInPlace(), f_inv);
  }

  BinaryField f_sqr = f.Square();
  EXPECT_EQ(f * f, f_sqr);
  f.SquareInPlace();
  EXPECT_EQ(f, f_sqr);

  BinaryField f_pow = f.Pow(5);
  EXPECT_EQ(f * f * f * f * f, f_pow);
}

}  // namespace tachyon::math
