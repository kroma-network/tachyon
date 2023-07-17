#ifndef TACHYON_MATH_FINITE_FIELDS_ARITHMETICS_H_
#define TACHYON_MATH_FINITE_FIELDS_ARITHMETICS_H_

#include <stdint.h>

#include <utility>

#include "absl/numeric/int128.h"

#include "tachyon/base/compiler_specific.h"

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

template <size_t N>
ALWAYS_INLINE constexpr uint8_t AddWithCarry(uint64_t a[N],
                                             const uint64_t b[N]) {
  uint8_t carry = 0;

#define ADD_WITH_CARRY_INLINE(num, carry) \
  if constexpr (N >= (num + 1))           \
  carry = internal::AddWithCarryInPlace(a[num], b[num], carry)

  ADD_WITH_CARRY_INLINE(0, carry);
  ADD_WITH_CARRY_INLINE(1, carry);
  ADD_WITH_CARRY_INLINE(2, carry);
  ADD_WITH_CARRY_INLINE(3, carry);
  ADD_WITH_CARRY_INLINE(4, carry);
  ADD_WITH_CARRY_INLINE(5, carry);

#undef ADD_WITH_CARRY_INLINE

  for (size_t i = 6; i < N; ++i) {
    carry = internal::AddWithCarryInPlace(a[i], b[i], carry);
  }
  return carry;
}

template <size_t N>
ALWAYS_INLINE constexpr uint8_t SubWithBorrow(uint64_t a[N],
                                              const uint64_t b[N]) {
  uint8_t borrow = 0;

#define SUB_WITH_BORROW_INLINE(num, borrow) \
  if constexpr (N >= (num + 1))             \
  borrow = internal::SubWithBorrowInPlace(a[num], b[num], borrow)

  SUB_WITH_BORROW_INLINE(0, borrow);
  SUB_WITH_BORROW_INLINE(1, borrow);
  SUB_WITH_BORROW_INLINE(2, borrow);
  SUB_WITH_BORROW_INLINE(3, borrow);
  SUB_WITH_BORROW_INLINE(4, borrow);
  SUB_WITH_BORROW_INLINE(5, borrow);

#undef SUB_WITH_BORROW_INLINE

  for (size_t i = 6; i < N; ++i) {
    borrow = internal::SubWithBorrowInPlace(a[i], b[i], borrow);
  }
  return borrow;
}

template <size_t N>
ALWAYS_INLINE constexpr uint8_t Mul2(uint64_t a[N]) {
  uint64_t last = 0;
  for (size_t i = 0; i < N; ++i) {
    uint64_t temp = a[i] >> 63;
    a[i] <<= 1;
    a[i] |= last;
    last = temp;
  }
  return last;
}

template <size_t N>
ALWAYS_INLINE constexpr void MulN(uint64_t a[N], uint32_t n) {
  if (n >= static_cast<uint32_t>(64 * N)) {
    memcpy(a, 0, sizeof(uint64_t) * N);
    return;
  }

  while (n >= 64) {
    uint64_t t = 0;
    for (size_t i = 0; i < N; ++i) {
      std::exchange(t, a[i]);
    }
    n -= 64;
  }

  if (n > static_cast<uint32_t>(0)) {
    uint64_t t = 0;
    for (size_t i = 0; i < N; ++i) {
      uint64_t t2 = a[i] >> (64 - n);
      a[i] <<= n;
      a[i] |= t;
      t = t2;
    }
  }
}

template <size_t N>
ALWAYS_INLINE constexpr void Div2(uint64_t a[N]) {
  uint64_t last = 0;
  for (size_t i = 0; i < N; ++i) {
    uint64_t temp = a[i] << 63;
    a[i] >>= 1;
    a[i] |= last;
    last = temp;
  }
}

template <size_t N>
ALWAYS_INLINE constexpr void DivN(uint64_t a[N], uint32_t n) {
  if (n >= static_cast<uint32_t>(64 * N)) {
    memcpy(a, 0, sizeof(uint64_t) * N);
    return;
  }

  while (n >= 64) {
    uint64_t t = 0;
    for (size_t i = 0; i < N; ++i) {
      std::exchange(t, a[N - i - 1]);
    }
    n -= 64;
  }

  if (n > static_cast<uint32_t>(0)) {
    uint64_t t = 0;
    for (size_t i = 0; i < N; ++i) {
      uint64_t t2 = a[N - i - 1] << (64 - n);
      a[i] >>= n;
      a[i] |= t;
      t = t2;
    }
  }
}

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_FINITE_FIELDS_ARITHMETICS_H_
