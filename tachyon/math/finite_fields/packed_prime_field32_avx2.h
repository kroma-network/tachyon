#ifndef TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX2_H_
#define TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX2_H_

#include <immintrin.h>

#include "tachyon/base/compiler_specific.h"

namespace tachyon::math {

ALWAYS_INLINE __m256i AddMod32(__m256i lhs, __m256i rhs, __m256i p) {
  // NOTE(chokobole): This assumes that the 2p - 2 < 2³², where p is modulus.
  // We want this to compile to:
  //      vpaddd   t, lhs, rhs
  //      vpsubd   u, t, p
  //      vpminud  r, t, u
  // throughput: 1 cyc/vec (8 els/cyc)
  // latency: 3 cyc

  // Let t := lhs + rhs
  //     u := (t - p) mod 2³²
  //     r := min(t, u)
  //
  // 0 ≤ lhs, rhs ≤ p - 1
  // 0 ≤ t ≤ 2p - 2
  //
  // 1) if 0 ≤ t ≤ p - 1:
  //    2³² - p ≤ u ≤ 2³² - 1
  //    2(p + 1) - p ≤ u ≤ 2³² - 1
  //    p - 1 < p + 1 ≤ u ≤ 2³² - 1
  //    r = t
  //
  // 2) otherwise p ≤ t ≤ 2p - 2:
  //    0 ≤ u ≤ p - 2 < p
  //    r = u
  //
  // In both cases, r is in {0, ..., p - 1}.
  __m256i t = _mm256_add_epi32(lhs, rhs);
  __m256i u = _mm256_sub_epi32(t, p);
  return _mm256_min_epu32(t, u);
}

ALWAYS_INLINE __m256i SubMod32(__m256i lhs, __m256i rhs, __m256i p) {
  // NOTE(chokobole): This assumes that the 2p - 2 < 2³², where p is modulus.
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
  //
  // 0 ≤ lhs, rhs ≤ p - 1
  // -p + 1 ≤ d ≤ p - 1
  //
  // 1) if 0 ≤ d ≤ p - 1:
  //    0 ≤ t ≤ p - 1
  //    p - 1 < p ≤ u ≤ 2p - 1
  //    r = t
  //
  // 2) otherwise -p + 1 ≤ d ≤ -1:
  //    2³² - p + 1 ≤ t ≤ 2³² - 1
  //    2(p + 1) - p + 1 ≤ t ≤ 2³² - 1
  //    p + 3 ≤ t ≤ 2³² - 1
  //    1 ≤ u ≤ p - 1 < p + 3
  //    r = u
  //
  // In both cases, r is in {0, ..., p - 1}.
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
  //                      { x            if y > 0,
  //     vpsignd(x, y) := { 0            if y = 0,
  //                      { -x mod 2³²   if y < 0.
  //
  // 0 ≤ val ≤ p - 1
  //
  // 1) if val = 0:
  //    r = 0
  //
  // 2) otherwise 1 ≤ val ≤ p - 1
  //    r = t
  //    2³² - p + 1 ≤ -val ≤ 2³² - 1
  //    1 ≤ t ≤ p - 1
  //
  // In both cases, r is in {0, ..., p - 1}.
  __m256i t = _mm256_sub_epi32(p, val);
  return _mm256_sign_epi32(t, val);
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX2_H_
