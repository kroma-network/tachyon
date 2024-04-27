#ifndef TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX512_H_
#define TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX512_H_

#include <immintrin.h>

#include "tachyon/base/compiler_specific.h"

namespace tachyon::math {

ALWAYS_INLINE __m512i AddMod32(__m512i lhs, __m512i rhs, __m512i p) {
  // NOTE(chokobole): This assumes 2p < 2³², where p is the modulus.
  // We want this to compile to:
  //      vpaddd   t, lhs, rhs
  //      vpsubd   u, t, p
  //      vpminud  r, t, u
  // throughput: 1.5 cyc/vec (10.67 els/cyc)
  // latency: 3 cyc

  // Let t := lhs + rhs
  //     u := (t - p) mod 2³²
  //     r := min(t, u)
  //     m := { p     if this is montgomery form, which is 0
  //          { p - 1 otherwise
  //
  // 0 ≤ lhs, rhs ≤ m
  //
  // 1) (lhs = p && rhs = 0) || (lhs = 0 && rhs = p)
  //    t = p
  //    u = 0
  //    r = 0
  //
  // 2) lhs = p && rhs = p
  //    t = 2p
  //    u = p
  //    r = p, which is 0.
  //
  // 3) (lhs = p && 1 ≤ rhs ≤ p - 1) || (1 ≤ lhs ≤ p - 1 && rhs = p)
  //    p + 1 ≤ t ≤ 2p - 1, go to 5)
  //
  // 4) 0 ≤ t ≤ p - 1
  //    p < 2³² - p ≤ u ≤ 2³² - 1
  //    r = t
  //
  // 5) p + 1 ≤ t ≤ 2p - 1
  //    1 ≤ u ≤ p - 1 ≤ p
  //    r = u
  //
  // In all cases, r is in {0, ..., m}.
  __m512i t = _mm512_add_epi32(lhs, rhs);
  __m512i u = _mm512_sub_epi32(t, p);
  return _mm512_min_epu32(t, u);
}

ALWAYS_INLINE __m512i SubMod32(__m512i lhs, __m512i rhs, __m512i p) {
  // NOTE(chokobole): This assumes 2p < 2³², where p is the modulus.
  // We want this to compile to:
  //      vpsubd   t, lhs, rhs
  //      vpaddd   u, t, p
  //      vpminud  r, t, u
  // throughput: 1.5 cyc/vec (10.67 els/cyc)
  // latency: 3 cyc

  // Let d := lhs - rhs
  //     t := d mod 2³²
  //     u := (t + p) mod 2³²
  //     r := min(t, u)
  //     m := { p     if this is montgomery form, which is 0
  //          { p - 1 otherwise
  //
  // 0 ≤ lhs, rhs ≤ m
  //
  // 1) lhs = p && rhs = 0
  //    d = p
  //    t = p
  //    r = p, which is 0.
  //
  // 2) lhs = 0 && rhs = p
  //    d = -p
  //    t = 2³² - p
  //    u = 0
  //    r = 0
  //
  // 3) lhs = p && rhs = p
  //    d = 0
  //    t = 0
  //    u = p
  //    r = 0
  //
  // 4) lhs = p && 1 ≤ rhs ≤ p - 1
  //    1 ≤ d ≤ p - 1, go to 6)
  //
  // 5) 1 ≤ lhs ≤ p - 1 && rhs = p
  //    -p + 1 ≤ d ≤ -1, go to 7)
  //
  // 6) 1 ≤ d ≤ p - 1
  //    1 ≤ t ≤ p - 1
  //    p + 1 ≤ u ≤ 2p - 1
  //    r = t
  //
  // 7) -p + 1 ≤ d ≤ -1
  //    2³² - p + 1 ≤ t ≤ 2³² - 1
  //    1 ≤ u ≤ p - 1
  //    r = u
  //
  // In all cases, r is in {0, ..., m}.
  __m512i t = _mm512_sub_epi32(lhs, rhs);
  __m512i u = _mm512_add_epi32(t, p);
  return _mm512_min_epu32(t, u);
}

ALWAYS_INLINE __m512i NegateMod32(__m512i val, __m512i p) {
  // We want this to compile to:
  //      vptestmd  nonzero, val, val
  //      vpsubd    res{nonzero}{z}, p, val
  // throughput: 1 cyc/vec (16 els/cyc)
  // latency: 4 cyc

  // NOTE: This routine prioritizes throughput over latency. An alternative
  // method would be to do sub(0, val), which would result in shorter
  // latency, but also lower throughput.

  // If |val| is zero, then the result is zeroed by masking.
  // Else if |val| is p, then the result is 0.
  // Otherwise |val| is in {1, ..., p - 1} and |p - val| is in the same range.
  __mmask16 nonzero = _mm512_test_epi32_mask(val, val);
  return _mm512_maskz_sub_epi32(nonzero, p, val);
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX512_H_
