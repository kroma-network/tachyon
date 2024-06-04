#include "tachyon/math/finite_fields/goldilocks/goldilocks_prime_field.h"

#include "absl/numeric/int128.h"
#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

namespace tachyon::math {

namespace {

uint64_t RandomForTesting() {
  return base::Uniform(
      base::Range<uint64_t>::Until(Goldilocks::Config::kModulus[0]));
}

}  // namespace

TEST(GoldilocksPrimeFieldTest, AdditiveOperators) {
  absl::uint128 M(Goldilocks::Config::kModulus[0]);

  absl::uint128 a(RandomForTesting());
  absl::uint128 b(RandomForTesting());
  SCOPED_TRACE(absl::Substitute("a: $0, b: $1", static_cast<uint64_t>(a),
                                static_cast<uint64_t>(b)));

  uint64_t sum = static_cast<uint64_t>((a + b) % M);
  uint64_t amb = static_cast<uint64_t>((a > b ? a - b : a + M - b) % M);
  uint64_t bma = static_cast<uint64_t>((b > a ? b - a : b + M - a) % M);

  Goldilocks fa = Goldilocks(static_cast<uint64_t>(a));
  Goldilocks fb = Goldilocks(static_cast<uint64_t>(b));

  EXPECT_EQ((fa + fb).ToBigInt()[0], sum);
  EXPECT_EQ((fb + fa).ToBigInt()[0], sum);
  EXPECT_EQ((fa - fb).ToBigInt()[0], amb);
  EXPECT_EQ((fb - fa).ToBigInt()[0], bma);

  Goldilocks tmp = fa;
  tmp += fb;
  EXPECT_EQ(tmp.ToBigInt()[0], sum);
  tmp -= fb;
  EXPECT_EQ(tmp.ToBigInt()[0], a);
}

TEST(GoldilocksPrimeFieldTest, MultiplicativeOperators) {
  absl::uint128 M = Goldilocks::Config::kModulus[0];

  absl::uint128 a(RandomForTesting());
  absl::uint128 b(RandomForTesting());
  SCOPED_TRACE(absl::Substitute("a: $0, b: $1", static_cast<uint64_t>(a),
                                static_cast<uint64_t>(b)));

  uint64_t mul = static_cast<uint64_t>((a * b) % M);

  Goldilocks fa = Goldilocks(static_cast<uint64_t>(a));
  Goldilocks fb = Goldilocks(static_cast<uint64_t>(b));

  EXPECT_EQ((fa * fb).ToBigInt()[0], mul);
  EXPECT_EQ((fb * fa).ToBigInt()[0], mul);

  Goldilocks tmp = fa;
  tmp *= fb;
  EXPECT_EQ((tmp).ToBigInt()[0], mul);
  if (UNLIKELY(fb.IsZero())) {
    ASSERT_FALSE(tmp /= fb);
  } else {
    ASSERT_TRUE(tmp /= fb);
  }
  EXPECT_EQ((tmp).ToBigInt()[0], a);
}

}  // namespace tachyon::math
