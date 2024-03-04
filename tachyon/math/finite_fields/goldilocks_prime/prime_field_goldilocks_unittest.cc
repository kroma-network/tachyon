#include "absl/numeric/int128.h"
#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/math/finite_fields/goldilocks_prime/goldilocks.h"

namespace tachyon::math {

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

}  // namespace tachyon::math
