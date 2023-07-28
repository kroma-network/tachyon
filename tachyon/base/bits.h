// Copyright 2013 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This file defines some bit utilities.

#ifndef TACHYON_BASE_BITS_H_
#define TACHYON_BASE_BITS_H_

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/logging.h"
#include "tachyon/build/build_config.h"

namespace tachyon::base {
namespace bits {

// Returns true iff |value| is a power of 2.
//
// TODO(pkasting): When C++20 is available, replace with std::has_single_bit().
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr bool IsPowerOfTwo(T value) {
  // From "Hacker's Delight": Section 2.1 Manipulating Rightmost Bits.
  //
  // Only positive integers with a single bit set are powers of two. If only one
  // bit is set in x (e.g. 0b00000100000000) then |x-1| will have that bit set
  // to zero and all bits to its right set to 1 (e.g. 0b00000011111111). Hence
  // |x & (x-1)| is 0 iff x is a power of two.
  return value > 0 && (value & (value - 1)) == 0;
}

// Round down |size| to a multiple of alignment, which must be a power of two.
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr T AlignDown(T size, T alignment) {
  DCHECK(IsPowerOfTwo(alignment));
  return size & ~(alignment - 1);
}

// Move |ptr| back to the previous multiple of alignment, which must be a power
// of two. Defined for types where sizeof(T) is one byte.
template <typename T, typename = typename std::enable_if<sizeof(T) == 1>::type>
inline T* AlignDown(T* ptr, uintptr_t alignment) {
  return reinterpret_cast<T*>(
      AlignDown(reinterpret_cast<uintptr_t>(ptr), alignment));
}

// Round up |size| to a multiple of alignment, which must be a power of two.
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
constexpr T AlignUp(T size, T alignment) {
  DCHECK(IsPowerOfTwo(alignment));
  return (size + alignment - 1) & ~(alignment - 1);
}

// Advance |ptr| to the next multiple of alignment, which must be a power of
// two. Defined for types where sizeof(T) is one byte.
template <typename T, typename = typename std::enable_if<sizeof(T) == 1>::type>
inline T* AlignUp(T* ptr, uintptr_t alignment) {
  return reinterpret_cast<T*>(
      AlignUp(reinterpret_cast<uintptr_t>(ptr), alignment));
}

// CountLeadingZeroBits(value) returns the number of zero bits following the
// most significant 1 bit in |value| if |value| is non-zero, otherwise it
// returns {sizeof(T) * 8}.
// Example: 00100010 -> 2
//
// CountTrailingZeroBits(value) returns the number of zero bits preceding the
// least significant 1 bit in |value| if |value| is non-zero, otherwise it
// returns {sizeof(T) * 8}.
// Example: 00100010 -> 1
//
// C does not have an operator to do this, but fortunately the various
// compilers have built-ins that map to fast underlying processor instructions.
//
// TODO(pkasting): When C++20 is available, replace with std::countl_zero() and
// similar.

// __builtin_clz has undefined behaviour for an input of 0, even though there's
// clearly a return value that makes sense, and even though some processor clz
// instructions have defined behaviour for 0. We could drop to raw __asm__ to
// do better, but we'll avoid doing that unless we see proof that we need to.
template <typename T, int bits = sizeof(T) * 8>
ALWAYS_INLINE constexpr
    typename std::enable_if<std::is_unsigned<T>::value && sizeof(T) <= 8,
                            int>::type
    CountLeadingZeroBits(T value) {
  static_assert(bits > 0, "invalid instantiation");
  return LIKELY(value)
             ? bits == 64
                   ? __builtin_clzll(static_cast<uint64_t>(value))
                   : __builtin_clz(static_cast<uint32_t>(value)) - (32 - bits)
             : bits;
}

template <typename T, int bits = sizeof(T) * 8>
ALWAYS_INLINE constexpr
    typename std::enable_if<std::is_unsigned<T>::value && sizeof(T) <= 8,
                            int>::type
    CountTrailingZeroBits(T value) {
  return LIKELY(value) ? bits == 64
                             ? __builtin_ctzll(static_cast<uint64_t>(value))
                             : __builtin_ctz(static_cast<uint32_t>(value))
                       : bits;
}

// Returns the integer i such as 2^i <= n < 2^(i+1).
//
// There is a common `BitLength` function, which returns the number of bits
// required to represent a value. Rather than implement that function,
// use `Log2Floor` and add 1 to the result.
//
// TODO(pkasting): When C++20 is available, replace with std::bit_xxx().
constexpr int Log2Floor(uint32_t n) { return 31 - CountLeadingZeroBits(n); }

// Returns the integer i such as 2^(i-1) < n <= 2^i.
constexpr int Log2Ceiling(uint32_t n) {
  // When n == 0, we want the function to return -1.
  // When n == 0, (n - 1) will underflow to 0xFFFFFFFF, which is
  // why the statement below starts with (n ? 32 : -1).
  return (n ? 32 : -1) - CountLeadingZeroBits(n - 1);
}

// Returns a value of type T with a single bit set in the left-most position.
// Can be used instead of manually shifting a 1 to the left.
template <typename T>
constexpr T LeftmostBit() {
  static_assert(std::is_integral<T>::value,
                "This function can only be used with integral types.");
  T one(1u);
  return one << (8 * sizeof(T) - 1);
}

}  // namespace bits
}  // namespace tachyon::base

#endif  // TACHYON_BASE_BITS_H_
