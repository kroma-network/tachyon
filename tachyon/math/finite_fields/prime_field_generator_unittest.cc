#include "gtest/gtest.h"

#include "tachyon/math/elliptic_curves/bls12/bls12_381/fq.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_381/fr.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/elliptic_curves/pasta/pallas/fq.h"
#include "tachyon/math/elliptic_curves/pasta/pallas/fr.h"
#include "tachyon/math/elliptic_curves/pasta/vesta/fq.h"
#include "tachyon/math/elliptic_curves/pasta/vesta/fr.h"
#include "tachyon/math/elliptic_curves/secp/secp256k1/fq.h"
#include "tachyon/math/elliptic_curves/secp/secp256k1/fr.h"
#include "tachyon/math/finite_fields/goldilocks/goldilocks_prime_field.h"
#include "tachyon/math/finite_fields/mersenne31/mersenne31.h"

namespace tachyon::math {

namespace {
template <typename PrimeField>
class PrimeFieldGeneratorTest : public testing::Test {
 public:
  static void SetUpTestSuite() { PrimeField::Init(); }
};

}  // namespace

using PrimeFieldTypes =
    testing::Types<bls12_381::Fq, bls12_381::Fr, bn254::Fq, bn254::Fr,
                   pallas::Fq, pallas::Fr, vesta::Fq, vesta::Fr, secp256k1::Fr,
                   secp256k1::Fq, Goldilocks, Mersenne31>;

TYPED_TEST_SUITE(PrimeFieldGeneratorTest, PrimeFieldTypes);

TYPED_TEST(PrimeFieldGeneratorTest, FromString) {
  using PrimeField = TypeParam;
  EXPECT_EQ(*PrimeField::FromDecString("3"), PrimeField(3));
  EXPECT_FALSE(PrimeField::FromDecString("x").has_value());
  EXPECT_EQ(*PrimeField::FromHexString("0x3"), PrimeField(3));
  EXPECT_FALSE(PrimeField::FromHexString("x").has_value());
}

TYPED_TEST(PrimeFieldGeneratorTest, ToString) {
  using PrimeField = TypeParam;
  PrimeField f(3);

  EXPECT_EQ(f.ToString(), "3");
  EXPECT_EQ(f.ToHexString(), "0x3");
}

TYPED_TEST(PrimeFieldGeneratorTest, Zero) {
  using PrimeField = TypeParam;
  EXPECT_TRUE(PrimeField::Zero().IsZero());
  EXPECT_FALSE(PrimeField::One().IsZero());
}

TYPED_TEST(PrimeFieldGeneratorTest, One) {
  using PrimeField = TypeParam;
  EXPECT_TRUE(PrimeField::One().IsOne());
  EXPECT_FALSE(PrimeField::Zero().IsOne());
  EXPECT_EQ(PrimeField::Config::kOne, PrimeField(1).ToMontgomery());
}

TYPED_TEST(PrimeFieldGeneratorTest, BigIntConversion) {
  using PrimeField = TypeParam;
  PrimeField r = PrimeField::Random();
  EXPECT_EQ(PrimeField::FromBigInt(r.ToBigInt()), r);
}

TYPED_TEST(PrimeFieldGeneratorTest, MontgomeryConversion) {
  using PrimeField = TypeParam;
  PrimeField r = PrimeField::Random();
  EXPECT_EQ(PrimeField::FromMontgomery(r.ToMontgomery()), r);
}

TYPED_TEST(PrimeFieldGeneratorTest, MpzClassConversion) {
  using PrimeField = TypeParam;
  PrimeField r = PrimeField::Random();
  EXPECT_EQ(PrimeField::FromMpzClass(r.ToMpzClass()), r);
}

TYPED_TEST(PrimeFieldGeneratorTest, EqualityOperators) {
  using PrimeField = TypeParam;
  PrimeField f(3);
  PrimeField f2(4);
  EXPECT_TRUE(f == f);
  EXPECT_TRUE(f != f2);
}

TYPED_TEST(PrimeFieldGeneratorTest, ComparisonOperator) {
  using PrimeField = TypeParam;
  PrimeField f(3);
  PrimeField f2(4);
  EXPECT_TRUE(f < f2);
  EXPECT_TRUE(f <= f2);
  EXPECT_FALSE(f > f2);
  EXPECT_FALSE(f >= f2);
}

TYPED_TEST(PrimeFieldGeneratorTest, AdditiveGroupOperators) {
  using PrimeField = TypeParam;
  PrimeField f = PrimeField::Random();
  SCOPED_TRACE(absl::Substitute("f: $0", f.ToString()));
  PrimeField f_neg = -f;
  EXPECT_TRUE((f_neg + f).IsZero());
  f.NegateInPlace();
  EXPECT_EQ(f, f_neg);

  PrimeField f_double = f.Double();
  EXPECT_EQ(f + f, f_double);
  f.DoubleInPlace();
  EXPECT_EQ(f, f_double);
}

TYPED_TEST(PrimeFieldGeneratorTest, MultiplicativeGroupOperators) {
  using PrimeField = TypeParam;
  PrimeField f = PrimeField::Random();
  SCOPED_TRACE(absl::Substitute("f: $0", f.ToString()));
  PrimeField f_inv = f.Inverse();
  EXPECT_EQ(f * f_inv, PrimeField::One());
  f.InverseInPlace();
  EXPECT_EQ(f, f_inv);

  PrimeField f_sqr = f.Square();
  EXPECT_EQ(f * f, f_sqr);
  f.SquareInPlace();
  EXPECT_EQ(f, f_sqr);

  PrimeField f_pow = f.Pow(5);
  EXPECT_EQ(f * f * f * f * f, f_pow);
}

}  // namespace tachyon::math
