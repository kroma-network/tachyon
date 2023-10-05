#include "tachyon/math/base/arithmetics.h"

#include <limits>

#include "absl/strings/substitute.h"
#include "gtest/gtest.h"

#include "tachyon/base/random.h"

namespace tachyon::math::internal {
namespace {

uint32_t GetRandomLo32() {
  return base::Uniform(uint32_t{0}, std::numeric_limits<uint32_t>::max() / 2);
}

uint32_t GetRandomHi32() {
  return base::Uniform(std::numeric_limits<uint32_t>::max() / 2,
                       std::numeric_limits<uint32_t>::max());
}

uint32_t GetRandomSqrt32() {
  return base::Uniform(
      uint32_t{0},
      static_cast<uint32_t>(std::sqrt(std::numeric_limits<uint32_t>::max())));
}

uint64_t GetRandomLo64() {
  return base::Uniform(uint64_t{0}, std::numeric_limits<uint64_t>::max() / 2);
}

uint64_t GetRandomHi64() {
  return base::Uniform(std::numeric_limits<uint64_t>::max() / 2,
                       std::numeric_limits<uint64_t>::max());
}

uint64_t GetRandomSqrt64() {
  return base::Uniform(
      uint64_t{0},
      static_cast<uint64_t>(std::sqrt(std::numeric_limits<uint64_t>::max())));
}

}  // namespace

TEST(Arithmetics, AddWithCarry32) {
  uint32_t a = GetRandomLo32();
  uint32_t b = GetRandomLo32();
  SCOPED_TRACE(absl::Substitute("a: $0, b: $1", a, b));

  auto result = u32::AddWithCarry(a, b);
  EXPECT_EQ(result.result, a + b);
  EXPECT_EQ(result.carry, 0);

  b = std::numeric_limits<uint32_t>::max() - a;
  auto result2 = u32::AddWithCarry(a, b, 1);
  EXPECT_EQ(result2.result, a + b + 1);
  EXPECT_EQ(result2.carry, 1);
}

TEST(Arithmetics, AddWithCarry64) {
  uint64_t a = GetRandomLo64();
  uint64_t b = GetRandomLo64();
  SCOPED_TRACE(absl::Substitute("a: $0, b: $1", a, b));

  auto result = u64::AddWithCarry(a, b);
  EXPECT_EQ(result.result, a + b);
  EXPECT_EQ(result.carry, 0);

  b = std::numeric_limits<uint64_t>::max() - a;
  auto result2 = u64::AddWithCarry(a, b, 1);
  EXPECT_EQ(result2.result, a + b + 1);
  EXPECT_EQ(result2.carry, 1);
}

TEST(Arithmetics, SubWithBorrow32) {
  uint32_t a = GetRandomHi32();
  uint32_t b = GetRandomLo32();
  SCOPED_TRACE(absl::Substitute("a: $0, b: $1", a, b));

  auto result = u32::SubWithBorrow(a, b);
  EXPECT_EQ(result.result, a - b);
  EXPECT_EQ(result.borrow, 0);

  auto result2 = u32::SubWithBorrow(b, a);
  EXPECT_EQ(result2.result, b - a);
  EXPECT_EQ(result2.borrow, 1);

  b = a;
  auto result3 = u32::SubWithBorrow(a, b, 1);
  EXPECT_EQ(result3.result, a - b - 1);
  EXPECT_EQ(result3.borrow, 1);
}

TEST(Arithmetics, SubWithBorrow64) {
  uint64_t a = GetRandomHi64();
  uint64_t b = GetRandomLo64();
  SCOPED_TRACE(absl::Substitute("a: $0, b: $1", a, b));

  auto result = u64::SubWithBorrow(a, b);
  EXPECT_EQ(result.result, a - b);
  EXPECT_EQ(result.borrow, 0);

  auto result2 = u64::SubWithBorrow(b, a);
  EXPECT_EQ(result2.result, b - a);
  EXPECT_EQ(result2.borrow, 1);

  b = a;
  auto result3 = u64::SubWithBorrow(a, b, 1);
  EXPECT_EQ(result3.result, a - b - 1);
  EXPECT_EQ(result3.borrow, 1);
}

TEST(Arithmetics, MulAddWithCarry32) {
  uint32_t a = GetRandomLo32();
  uint32_t b = GetRandomSqrt32() / 2;
  uint32_t c = GetRandomSqrt32();
  SCOPED_TRACE(absl::Substitute("a: $0, b: $1, c: $2", a, b, c));

  uint64_t bc = uint64_t{b} * uint64_t{c};
  auto result = u32::MulAddWithCarry(a, b, c);
  EXPECT_EQ(result.lo, static_cast<uint32_t>(uint64_t{a} + bc));
  EXPECT_EQ(result.hi, 0);

  a = static_cast<uint32_t>(std::numeric_limits<uint64_t>::max() - bc);
  auto result2 = u32::MulAddWithCarry(a, b, c, 1);
  EXPECT_EQ(result2.lo, static_cast<uint32_t>(uint64_t{a} + bc + uint64_t{1}));
  EXPECT_EQ(result2.hi, 1);
}

TEST(Arithmetics, MulAddWithCarry64) {
  uint64_t a = GetRandomLo64();
  uint64_t b = GetRandomSqrt64() / 2;
  uint64_t c = GetRandomSqrt64();
  SCOPED_TRACE(absl::Substitute("a: $0, b: $1, c: $2", a, b, c));

  absl::uint128 bc = absl::uint128(b) * absl::uint128(c);
  auto result = u64::MulAddWithCarry(a, b, c);
  EXPECT_EQ(result.lo, static_cast<uint64_t>(absl::uint128(a) + bc));
  EXPECT_EQ(result.hi, 0);

  a = static_cast<uint64_t>(std::numeric_limits<absl::uint128>::max() - bc);
  auto result2 = u64::MulAddWithCarry(a, b, c, 1);
  EXPECT_EQ(result2.lo,
            static_cast<uint64_t>(absl::uint128(a) + bc + absl::uint128(1)));
  EXPECT_EQ(result2.hi, 1);
}

}  // namespace tachyon::math::internal
