#ifndef TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX2_H_
#define TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX2_H_

#include <immintrin.h>

#include "tachyon/base/compiler_specific.h"

namespace tachyon::math {

ALWAYS_INLINE __m256i AddMod32(__m256i lhs, __m256i rhs, __m256i p) {
  // NOTE(chokobole): This assumes 2p < 2³², where p is the modulus.
  // We want this to compile to:
  //      vpaddd   t, lhs, rhs
  //      vpsubd   u, t, p
  //      vpminud  r, t, u
  // throughput: 1 cyc/vec (8 els/cyc)
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
  __m256i t = _mm256_add_epi32(lhs, rhs);
  __m256i u = _mm256_sub_epi32(t, p);
  return _mm256_min_epu32(t, u);
}

ALWAYS_INLINE __m256i SubMod32(__m256i lhs, __m256i rhs, __m256i p) {
  // NOTE(chokobole): This assumes 2p < 2³², where p is the modulus.
  // We want this to compile to:
  //      vpsubd   t, lhs, rhs
  //      vpaddd   u, t, p
  //      vpminud  r, t, u
  // throughput: 1 cyc/vec (8 els/cyc)
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
  __m256i t = _mm256_sub_epi32(lhs, rhs);
  __m256i u = _mm256_add_epi32(t, p);
  return _mm256_min_epu32(t, u);
}

ALWAYS_INLINE __m256i NegateMod32(__m256i val, __m256i p) {
  // We want this to compile to:
  //      vpsubd   t, p, val
  //      vpsignd  r, t, val
  // throughput: .67 cyc/vec (12 els/cyc)
  // latency: 2 cyc

  // Let t := (p - val) mod 2³²
  //     r := vpsignd(t, val)
  //                      { x            if y > 0
  //     vpsignd(x, y) := { 0            if y = 0
  //                      { -x mod 2³²   if y < 0
  //     m := { p     if this is montgomery form, which is 0
  //          { p - 1 otherwise
  //
  // 0 ≤ val ≤ m
  //
  // 1) val = 0
  //    r = 0
  //
  // 2) val = p
  //    r = t
  //    t = 0
  //
  // 3) 1 ≤ val ≤ p - 1
  //    r = t
  //    2³² - p + 1 ≤ -val ≤ 2³² - 1
  //    1 ≤ t ≤ p - 1
  //
  // In all cases, r is in {0, ..., m}.
  __m256i t = _mm256_sub_epi32(p, val);
  return _mm256_sign_epi32(t, val);
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX2_H_
