#ifndef TACHYON_MATH_BASE_ARITHMETICS_H_
#define TACHYON_MATH_BASE_ARITHMETICS_H_

#include <stdint.h>

#include <utility>

#include "absl/numeric/int128.h"

#include "tachyon/base/compiler_specific.h"
#include "tachyon/build/build_config.h"

namespace tachyon {
namespace math {
namespace internal {

// Set a = a + b + carry, and returns the new carry.
ALWAYS_INLINE constexpr uint8_t AddWithCarryInPlace(uint64_t& a, uint64_t b,
                                                    uint8_t carry) {
  absl::uint128 tmp =
      absl::uint128(a) + absl::uint128(b) + absl::uint128(carry);
  a = static_cast<uint64_t>(tmp);
  return static_cast<uint8_t>(tmp >> 64);
}

// Calculate a + b + carry, returning the sum.
ALWAYS_INLINE constexpr uint64_t AddWithCarry(uint64_t a, uint64_t b,
                                              uint8_t carry) {
  absl::uint128 tmp =
      absl::uint128(a) + absl::uint128(b) + absl::uint128(carry);
  return static_cast<uint64_t>(tmp);
}

// Set a = a - b - borrow, and returns the borrow.
ALWAYS_INLINE constexpr uint8_t SubWithBorrowInPlace(uint64_t& a, uint64_t b,
                                                     uint8_t borrow) {
  absl::uint128 tmp = (absl::uint128(1) << 64) + absl::uint128(a) -
                      absl::uint128(b) - absl::uint128(borrow);
  a = static_cast<uint64_t>(tmp);
  return static_cast<uint8_t>(tmp >> 64) == 0;
}

// Calculate a + b * c, returning the lower 64 bits of the result and setting
// `carry` to the upper 64 bits.
// NOTE(chokobole): cannot be marked with constexpr because of absl::uint128
// multiplication.
ALWAYS_INLINE uint64_t MulAdd(uint64_t& a, uint64_t b, uint64_t c,
                              uint64_t& carry) {
  absl::uint128 tmp = absl::uint128(a) + absl::uint128(b) * absl::uint128(c);
  carry = static_cast<uint64_t>(tmp >> 64);
  return static_cast<uint64_t>(tmp);
}

// Calculate a + (b * c) + carry, returning the lower 64 bits of the result
// and setting `carry` to the upper 64 bits.
// NOTE(chokobole): cannot be marked with constexpr because of absl::uint128
// multiplication.
ALWAYS_INLINE uint64_t MulAddWithCarry(uint64_t& a, uint64_t b, uint64_t c,
                                       uint64_t& carry) {
  absl::uint128 tmp = absl::uint128(a) + absl::uint128(b) * absl::uint128(c) +
                      absl::uint128(carry);
  carry = static_cast<uint64_t>(tmp >> 64);
  return static_cast<uint64_t>(tmp);
}

}  // namespace internal
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_ARITHMETICS_H_
