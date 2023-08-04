#include "absl/numeric/int128.h"
#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/goldilocks_prime/goldilocks.h"

namespace tachyon::math {

TEST(PrimeFieldGoldilocksTest, FromString) {
  EXPECT_EQ(Goldilocks::FromDecString("3"), Goldilocks(3));
  EXPECT_EQ(Goldilocks::FromHexString("0x3"), Goldilocks(3));
}

TEST(PrimeFieldGoldilocksTest, ToString) {
  Goldilocks f(3);

  EXPECT_EQ(f.ToString(), "3");
  EXPECT_EQ(f.ToHexString(), "0x3");
}

TEST(PrimeFieldGoldilocksTest, Zero) {
  EXPECT_TRUE(Goldilocks::Zero().IsZero());
  EXPECT_FALSE(Goldilocks::One().IsZero());
}

TEST(PrimeFieldGoldilocksTest, One) {
  EXPECT_TRUE(Goldilocks::One().IsOne());
  EXPECT_FALSE(Goldilocks::Zero().IsOne());
  EXPECT_EQ(Goldilocks::Config::kOne, Goldilocks(1).ToMontgomery());
}

TEST(PrimeFieldGoldilocksTest, BigIntConversion) {
  Goldilocks r = Goldilocks::Random();
  EXPECT_EQ(Goldilocks::FromBigInt(r.ToBigInt()), r);
}

TEST(PrimeFieldGoldilocksTest, MontgomeryConversion) {
  Goldilocks r = Goldilocks::Random();
  EXPECT_EQ(Goldilocks::FromMontgomery(r.ToMontgomery()), r);
}

TEST(PrimeFieldGoldilocksTest, MpzClassConversion) {
  Goldilocks r = Goldilocks::Random();
  EXPECT_EQ(Goldilocks::FromMpzClass(r.ToMpzClass()), r);
}

TEST(PrimeFieldGoldilocksTest, EqualityOperators) {
  Goldilocks f(3);
  Goldilocks f2(4);
  EXPECT_TRUE(f == f);
  EXPECT_TRUE(f != f2);
}

TEST(PrimeFieldGoldilocksTest, ComparisonOperator) {
  Goldilocks f(3);
  Goldilocks f2(4);
  EXPECT_TRUE(f < f2);
  EXPECT_TRUE(f <= f2);
  EXPECT_FALSE(f > f2);
  EXPECT_FALSE(f >= f2);
}

TEST(PrimeFieldGoldilocksTest, AdditiveOperators) {
  absl::uint128 M(Goldilocks::Config::kModulus[0]);

  absl::uint128 a(Goldilocks::RandomForTesting());
  absl::uint128 b(Goldilocks::RandomForTesting());
  SCOPED_TRACE(absl::Substitute("a: $0, b: $1", static_cast<uint64_t>(a),
                                static_cast<uint64_t>(b)));

  uint64_t sum = static_cast<uint64_t>((a + b) % M);
  uint64_t amb = static_cast<uint64_t>((a > b ? a - b : a + M - b) % M);
  uint64_t bma = static_cast<uint64_t>((b > a ? b - a : b + M - a) % M);

  Goldilocks fa = Goldilocks(static_cast<uint64_t>(a));
  Goldilocks fb = Goldilocks(static_cast<uint64_t>(b));

  EXPECT_EQ(static_cast<uint64_t>(fa + fb), sum);
  EXPECT_EQ(static_cast<uint64_t>(fb + fa), sum);
  EXPECT_EQ(static_cast<uint64_t>(fa - fb), amb);
  EXPECT_EQ(static_cast<uint64_t>(fb - fa), bma);

  Goldilocks tmp = fa;
  tmp += fb;
  EXPECT_EQ(static_cast<uint64_t>(tmp), sum);
  tmp -= fb;
  EXPECT_EQ(static_cast<uint64_t>(tmp), a);
}

TEST(PrimeFieldGoldilocksTest, AdditiveGroupOperators) {
  Goldilocks f = Goldilocks::Random();
  SCOPED_TRACE(absl::Substitute("f: $0", f.ToString()));
  Goldilocks f_neg = f.Negative();
  EXPECT_TRUE((f_neg + f).IsZero());
  f.NegInPlace();
  EXPECT_EQ(f, f_neg);

  Goldilocks f_double = f.Double();
  EXPECT_EQ(f + f, f_double);
  f.DoubleInPlace();
  EXPECT_EQ(f, f_double);
}

TEST(PrimeFieldGoldilocksTest, MultiplicativeOperators) {
  absl::uint128 M = Goldilocks::Config::kModulus[0];

  absl::uint128 a(Goldilocks::RandomForTesting());
  absl::uint128 b(Goldilocks::RandomForTesting());
  SCOPED_TRACE(absl::Substitute("a: $0, b: $1", static_cast<uint64_t>(a),
                                static_cast<uint64_t>(b)));

  uint64_t mul = static_cast<uint64_t>((a * b) % M);

  Goldilocks fa = Goldilocks(static_cast<uint64_t>(a));
  Goldilocks fb = Goldilocks(static_cast<uint64_t>(b));

  EXPECT_EQ(static_cast<uint64_t>(fa * fb), mul);
  EXPECT_EQ(static_cast<uint64_t>(fb * fa), mul);

  Goldilocks tmp = fa;
  tmp *= fb;
  EXPECT_EQ(static_cast<uint64_t>(tmp), mul);
  tmp /= fb;
  EXPECT_EQ(static_cast<uint64_t>(tmp), a);
}

TEST(PrimeFieldGoldilocksTest, MultiplicativeGroupOperators) {
  Goldilocks f = Goldilocks::Random();
  SCOPED_TRACE(absl::Substitute("f: $0", f.ToString()));
  Goldilocks f_inv = f.Inverse();
  EXPECT_EQ(f * f_inv, Goldilocks::One());
  f.InverseInPlace();
  EXPECT_EQ(f, f_inv);

  Goldilocks f_sqr = f.Square();
  EXPECT_EQ(f * f, f_sqr);
  f.SquareInPlace();
  EXPECT_EQ(f, f_sqr);

  Goldilocks f_pow = f.Pow(Goldilocks(5).ToBigInt());
  EXPECT_EQ(f * f * f * f * f, f_pow);
}

}  // namespace tachyon::math
