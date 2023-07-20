#ifndef TACHYON_MATH_BASE_ARITHMETICS_H_
#define TACHYON_MATH_BASE_ARITHMETICS_H_

#include <stdint.h>

#include <utility>

#include "absl/numeric/int128.h"

#include "tachyon/base/compiler_specific.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/base/arithmetics_results.h"

namespace tachyon {
namespace math {
namespace internal {

// Calculates a + b + carry.
ALWAYS_INLINE constexpr AddResult<uint64_t> AddWithCarry(uint64_t a, uint64_t b,
                                                         uint64_t carry = 0) {
  absl::uint128 tmp =
      absl::uint128(a) + absl::uint128(b) + absl::uint128(carry);
  AddResult<uint64_t> result;
  result.result = static_cast<uint64_t>(tmp);
  result.carry = static_cast<uint64_t>(tmp >> 64);
  return result;
}

// Calculates a - b - borrow.
ALWAYS_INLINE constexpr SubResult<uint64_t> SubWithBorrow(uint64_t& a,
                                                          uint64_t b,
                                                          uint64_t borrow = 0) {
  absl::uint128 tmp = (absl::uint128(1) << 64) + absl::uint128(a) -
                      absl::uint128(b) - absl::uint128(borrow);
  SubResult<uint64_t> result;
  result.result = static_cast<uint64_t>(tmp);
  result.borrow =
      static_cast<uint64_t>(static_cast<uint64_t>((tmp >> 64) == 0 ? 1 : 0));
  return result;
}

// Calculates a + b * c.
// NOTE(chokobole): cannot be marked with constexpr because of absl::uint128
// multiplication.
ALWAYS_INLINE MulResult<uint64_t> MulAddWithCarry(uint64_t a, uint64_t b,
                                                  uint64_t c,
                                                  uint64_t carry = 0) {
  absl::uint128 tmp = absl::uint128(a) + absl::uint128(b) * absl::uint128(c) + carry;
  MulResult<uint64_t> result;
  result.lo = static_cast<uint64_t>(tmp);
  result.hi = static_cast<uint64_t>(tmp >> 64);
  return result;
}

}  // namespace internal
}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_BASE_ARITHMETICS_H_
