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

// MONTGOMERY MULTIPLICATION
//   This implementation is based on [1] but with minor changes. The reduction
//   is as follows:
//
// Constants: P = modulus of the prime field
//            B = 2³²
//            μ = P⁻¹ mod B
// Input: 0 ≤ C < P * B
// Output: 0 ≤ R < P such that R = C * B⁻¹ (mod P)
//   1. Q := μ * C mod B
//   2. D := (C - Q * P) / B
//   3. R := if D < 0 then D + P else D
//
// We first show that the division in step 2. is exact. It suffices to show that
// C = Q * P (mod B). By definition of Q and μ, we have
// Q * P = μ * C * P = P⁻¹ * C * P = C (mod B).
// We also have C - Q * P = C (mod P), so thus D = C * B⁻¹ (mod P).
//
// It remains to show that R is in the correct range. It suffices to show that
// -P ≤ D < P. We know that 0 ≤ C < P * B and 0 ≤ Q * P < P * B.
// Then -P * B < C - Q * P < P * B and -P < D < P, as desired.
//
// [1] Modern Computer Arithmetic, Richard Brent and Paul Zimmermann,
//     Cambridge University Press, 2010, algorithm 2.7.
ALWAYS_INLINE __m256i MontyD(__m256i lhs, __m256i rhs, __m256i p, __m256i inv) {
  __m256i prod = _mm256_mul_epu32(lhs, rhs);
  __m256i q = _mm256_mul_epu32(prod, inv);
  __m256i q_p = _mm256_mul_epu32(q, p);
  return _mm256_sub_epi64(prod, q_p);
}

ALWAYS_INLINE __m256i movehdup_epi32(__m256i x) {
  // This instruction is only available in the floating-point flavor; this
  // distinction is only for historical reasons and no longer matters. We cast
  // to floats, duplicate, and cast back.
  return _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)));
}

ALWAYS_INLINE __m256i MontMulMod32(__m256i lhs, __m256i rhs, __m256i p,
                                   __m256i inv) {
  // We want this to compile to:
  //      vmovshdup  lhs_odd, lhs
  //      vmovshdup  rhs_odd, rhs
  //      vpmuludq   prod_evn, lhs, rhs
  //      vpmuludq   prod_odd, lhs_odd, rhs_odd
  //      vpmuludq   q_evn, prod_evn, inv
  //      vpmuludq   q_odd, prod_odd, inv
  //      vpmuludq   q_p_evn, q_evn, p
  //      vpmuludq   q_p_odd, q_odd, p
  //      vpsubq     d_evn, prod_evn, q_p_evn
  //      vpsubq     d_odd, prod_odd, q_p_odd
  //      vmovshdup  d_evn_hi, d_evn
  //      vpblendd   t, d_evn_hi, d_odd, aah
  //      vpaddd     u, t, p
  //      vpminud    r, t, u
  // throughput: 4.67 cyc/vec (1.71 els/cyc)
  // latency: 21 cyc

  __m256i lhs_evn = lhs;
  __m256i rhs_evn = rhs;
  __m256i lhs_odd = movehdup_epi32(lhs);
  __m256i rhs_odd = movehdup_epi32(rhs);

  __m256i d_evn = MontyD(lhs_evn, rhs_evn, p, inv);
  __m256i d_odd = MontyD(lhs_odd, rhs_odd, p, inv);

  __m256i d_evn_hi = movehdup_epi32(d_evn);
  __m256i t = _mm256_blend_epi32(d_evn_hi, d_odd, 0b10101010);

  __m256i u = _mm256_add_epi32(t, p);
  return _mm256_min_epu32(t, u);
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX2_H_
