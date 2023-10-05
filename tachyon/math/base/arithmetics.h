#ifndef TACHYON_MATH_BASE_ARITHMETICS_H_
#define TACHYON_MATH_BASE_ARITHMETICS_H_

#include <stdint.h>

#include <utility>

#include "absl/numeric/int128.h"

#include "tachyon/base/bit_cast.h"
#include "tachyon/base/compiler_specific.h"
#include "tachyon/build/build_config.h"
#include "tachyon/math/base/arithmetics_results.h"

#if defined(__clang__) && HAS_BUILTIN(__builtin_addc)
// See
// https://clang.llvm.org/docs/LanguageExtensions.html#multiprecision-arithmetic-builtins
#elif ARCH_CPU_X86_64
// See
// https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/intrinsics-for-multi-precision-arithmetic.html
#include <x86gprintrin.h>
#endif

namespace tachyon::math::internal {
namespace u32 {

// Calculates a + b + carry.
ALWAYS_INLINE AddResult<uint32_t> AddWithCarry(uint32_t a, uint32_t b,
                                               uint32_t carry = 0) {
  AddResult<uint32_t> result;
#if defined(__clang__) && HAS_BUILTIN(__builtin_addc)
  result.result = __builtin_addc(a, b, carry, &result.carry);
#elif ARCH_CPU_X86_64
  result.carry = _addcarry_u32(carry, a, b, &result.result);
#else
  uint64_t tmp = uint64_t{a} + uint64_t{b} + uint64_t{carry};
  result.result = static_cast<uint32_t>(tmp);
  result.carry = static_cast<uint32_t>(tmp >> 32);
#endif
  return result;
}

// Calculates a - b - borrow.
ALWAYS_INLINE SubResult<uint32_t> SubWithBorrow(uint32_t a, uint32_t b,
                                                uint32_t borrow = 0) {
  SubResult<uint32_t> result;
#if defined(__clang__) && HAS_BUILTIN(__builtin_subc)
  result.result = __builtin_subc(a, b, borrow, &result.borrow);
#elif ARCH_CPU_X86_64
  result.borrow = _subborrow_u32(borrow, a, b, &result.result);
#else
  uint64_t tmp =
      (uint64_t{1} << 32) + uint64_t{a} - uint64_t{b} - uint64_t{borrow};
  result.result = static_cast<uint32_t>(tmp);
  result.borrow =
      static_cast<uint32_t>(static_cast<uint32_t>((tmp >> 32) == 0 ? 1 : 0));
#endif
  return result;
}

// Calculates a + b * c.
ALWAYS_INLINE constexpr MulResult<uint32_t> MulAddWithCarry(
    uint32_t a, uint32_t b, uint32_t c, uint32_t carry = 0) {
  uint64_t tmp = uint64_t{a} + uint64_t{b} * uint64_t{c} + uint64_t{carry};
  MulResult<uint32_t> result;
  result.lo = static_cast<uint32_t>(tmp);
  result.hi = static_cast<uint32_t>(tmp >> 32);
  return result;
}

}  // namespace u32

namespace u64 {

// Calculates a + b + carry.
ALWAYS_INLINE AddResult<uint64_t> AddWithCarry(uint64_t a, uint64_t b,
                                               uint64_t carry = 0) {
  AddResult<uint64_t> result;
#if defined(__clang__) && HAS_BUILTIN(__builtin_addcl)
  unsigned long carry_result;  // NOLINT(runtime/int)
  result.result = __builtin_addcl(a, b, carry, &carry_result);
  result.carry = base::bit_cast<uint64_t>(carry_result);
#elif ARCH_CPU_X86_64
  unsigned long long add_result;  // NOLINT(runtime/int)
  result.carry = _addcarry_u64(carry, a, b, &add_result);
  result.result = base::bit_cast<uint64_t>(add_result);
#else
  absl::uint128 tmp =
      absl::uint128(a) + absl::uint128(b) + absl::uint128(carry);
  result.result = static_cast<uint64_t>(tmp);
  result.carry = static_cast<uint64_t>(tmp >> 64);
#endif
  return result;
}

// Calculates a - b - borrow.
ALWAYS_INLINE SubResult<uint64_t> SubWithBorrow(uint64_t& a, uint64_t b,
                                                uint64_t borrow = 0) {
  SubResult<uint64_t> result;
#if defined(__clang__) && HAS_BUILTIN(__builtin_subcl)
  unsigned long borrow_result;  // NOLINT(runtime/int)
  result.result = __builtin_subcl(a, b, borrow, &borrow_result);
  result.borrow = base::bit_cast<uint64_t>(borrow_result);
#elif ARCH_CPU_X86_64
  unsigned long long sub_result;  // NOLINT(runtime/int)
  result.borrow = _subborrow_u64(borrow, a, b, &sub_result);
  result.result = base::bit_cast<uint64_t>(sub_result);
#else
  absl::uint128 tmp = (absl::uint128(1) << 64) + absl::uint128(a) -
                      absl::uint128(b) - absl::uint128(borrow);
  result.result = static_cast<uint64_t>(tmp);
  result.borrow =
      static_cast<uint64_t>(static_cast<uint64_t>((tmp >> 64) == 0 ? 1 : 0));
#endif
  return result;
}

// Calculates a + b * c.
ALWAYS_INLINE MulResult<uint64_t> MulAddWithCarry(uint64_t a, uint64_t b,
                                                  uint64_t c,
                                                  uint64_t carry = 0) {
  absl::uint128 tmp = absl::uint128(a) + absl::uint128(b) * absl::uint128(c) +
                      absl::uint128(carry);
  MulResult<uint64_t> result;
  result.lo = static_cast<uint64_t>(tmp);
  result.hi = static_cast<uint64_t>(tmp >> 64);
  return result;
}

}  // namespace u64
}  // namespace tachyon::math::internal

#endif  // TACHYON_MATH_BASE_ARITHMETICS_H_
