// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

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

ALWAYS_INLINE __m512i movehdup_epi32(__m512i a) {
  // The instruction is only available in the floating-point flavor; this
  // distinction is only for historical reasons and no longer matters. We cast
  // to floats, do the thing, and cast back.
  return _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(a)));
}

// Viewing |a| as a vector of 16 |uint32_t|s, copy the odd elements into the
// even elements below them, then merge with |src| according to the mask
// provided. In other words, for all 0 ≤ i < 8, set the even elements
// according to
// |res[2 * i] := if k[2 * i] { a[2 * i + 1] } else { src[2 * i] }|, and
// the odd elements according to
// |res[2 * i + 1] := if k[2 * i + 1] { a[2 * i + 1] } else { src[2 * i + 1] }|.
ALWAYS_INLINE __m512i mask_movehdup_epi32(__m512i src, __mmask16 k, __m512i a) {
  // The instruction is only available in the floating-point flavor; this
  // distinction is only for historical reasons and no longer matters. We cast
  // to floats, do the thing, and cast back.
  return _mm512_castps_si512(_mm512_mask_movehdup_ps(
      _mm512_castsi512_ps(src), k, _mm512_castsi512_ps(a)));
}

ALWAYS_INLINE __m512i mask_moveldup_epi32(__m512i src, __mmask16 k, __m512i a) {
  // The instruction is only available in the floating-point flavor; this
  // distinction is only for historical reasons and no longer matters. We cast
  // to floats, do the thing, and cast back.
  return _mm512_castps_si512(_mm512_mask_moveldup_ps(
      _mm512_castsi512_ps(src), k, _mm512_castsi512_ps(a)));
}

ALWAYS_INLINE __m512i MontMulMod32(__m512i lhs, __m512i rhs, __m512i p,
                                   __m512i inv) {
  constexpr __mmask16 kEvens = 0b0101010101010101;

  // We want this to compile to:
  //      vmovshdup  lhs_odd, lhs
  //      vmovshdup  rhs_odd, rhs
  //      vpmuludq   prod_evn, lhs, rhs
  //      vpmuludq   prod_hi, lhs_odd, rhs_odd
  //      vpmuludq   q_evn, prod_evn, inv
  //      vpmuludq   q_odd, prod_hi, inv
  //      vmovshdup  prod_hi{kEvens}, prod_evn
  //      vpmuludq   q_p_evn, q_evn, p
  //      vpmuludq   q_p_hi, q_odd, p
  //      vmovshdup  q_p_hi{kEvens}, q_p_evn
  //      vpcmpltud  underflow, prod_hi, q_p_hi
  //      vpsubd     r, prod_hi, q_p_hi
  //      vpaddd     r{underflow}, r, p
  // throughput: 6.5 cyc/vec (2.46 els/cyc)
  // latency: 21 cyc

  // |vpmuludq| only reads the even doublewords, so when we pass |lhs| and
  // |rhs| directly we get the eight products at even positions.
  //            ┌──────┬──────┬─────┬───────┬───────┐
  // |lhs_evn|: │ lhs₀ │ lhs₁ │ ... │ lhs₁₄ │ lhs₁₅ │
  //            └──────┴──────┴─────┴───────┴───────┘
  __m512i lhs_evn = lhs;
  //            ┌──────┬──────┬─────┬───────┬───────┐
  // |rhs_evn|: │ rhs₀ │ rhs₁ │ ... │ rhs₁₄ │ rhs₁₅ │
  //            └──────┴──────┴─────┴───────┴───────┘
  __m512i rhs_evn = rhs;

  // Copy the odd doublewords into even positions to compute the eight
  // products at odd positions. NB: The odd doublewords are ignored by
  // |vpmuludq|, so we have a lot of choices for how to do this; |vmovshdup|
  // is nice because it runs on a memory port if the operand is in memory,
  // thus improving our throughput.
  //            ┌──────┬──────┬─────┬───────┬───────┐
  // |lhs_odd|: │ lhs₁ │ lhs₁ │ ... │ lhs₁₅ │ lhs₁₅ │
  //            └──────┴──────┴─────┴───────┴───────┘
  __m512i lhs_odd = movehdup_epi32(lhs);
  //            ┌──────┬──────┬─────┬───────┬───────┐
  // |rhs_odd|: │ rhs₁ │ rhs₁ │ ... │ rhs₁₅ │ rhs₁₅ │
  //            └──────┴──────┴─────┴───────┴───────┘
  __m512i rhs_odd = movehdup_epi32(rhs);

  // clang-format off
  //             ┌─────────────────┬─────────────────┬─────┬───────────────────┬───────────────────┐
  // |prod_evn|: │ Lo(lhs₀ * rhs₀) │ Hi(lhs₀ * rhs₀) | ... │ Lo(lhs₁₄ * rhs₁₄) │ Hi(lhs₁₄ * rhs₁₄) |
  //             └─────────────────┴─────────────────┴─────┴───────────────────┴───────────────────┘
  // clang-format on
  __m512i prod_evn = _mm512_mul_epu32(lhs_evn, rhs_evn);
  // clang-format off
  //             ┌─────────────────┬─────────────────┬─────┬───────────────────┬───────────────────┐
  // |prod_odd|: │ Lo(lhs₁ * rhs₁) │ Hi(lhs₁ * rhs₁) | ... │ Lo(lhs₁₅ * rhs₁₅) │ Hi(lhs₁₅ * rhs₁₅) |
  //             └─────────────────┴─────────────────┴─────┴───────────────────┴───────────────────┘
  // clang-format on
  __m512i prod_odd = _mm512_mul_epu32(lhs_odd, rhs_odd);

  // clang-format off
  //          ┌───────────────────────┬───────────────────────┬─────┬─────────────────────────┬─────────────────────────┐
  // |q_evn|: │ Lo(lhs₀ * rhs₀ * inv) │ Hi(lhs₀ * rhs₀ * inv) | ... │ Lo(lhs₁₄ * rhs₁₄ * inv) │ Hi(lhs₁₄ * rhs₁₄ * inv) |
  //          └───────────────────────┴───────────────────────┴─────┴─────────────────────────┴─────────────────────────┘
  // clang-format on
  __m512i q_evn = _mm512_mul_epu32(prod_evn, inv);
  // clang-format off
  //          ┌───────────────────────┬───────────────────────┬─────┬─────────────────────────┬─────────────────────────┐
  // |q_odd|: │ Lo(lhs₁ * rhs₁ * inv) │ Hi(lhs₁ * rhs₁ * inv) | ... │ Lo(lhs₁₅ * rhs₁₅ * inv) │ Hi(lhs₁₅ * rhs₁₅ * inv) |
  //          └───────────────────────┴───────────────────────┴─────┴─────────────────────────┴─────────────────────────┘
  // clang-format on
  __m512i q_odd = _mm512_mul_epu32(prod_odd, inv);

  // TODO(chokobole): Add diagram for better explanation.
  // Get all the high halves as one vector: this is |(lhs * rhs) >> 32|.
  // NB: |vpermt2d| may feel like a more intuitive choice here, but it has
  // much higher latency.
  __m512i prod_hi = mask_movehdup_epi32(prod_odd, kEvens, prod_evn);

  // Normally we'd want to mask to perform % 2**32, but the instruction below
  // only reads the low 32 bits anyway.
  // clang-format off
  //            ┌───────────────────────────┬───────────────────────────┬─────┬─────────────────────────────┬─────────────────────────────┐
  // |q_p_evn|: │ Lo(lhs₀ * rhs₀ * inv * p) │ Hi(lhs₀ * rhs₀ * inv * p) | ... │ Lo(lhs₁₄ * rhs₁₄ * inv * p) │ Hi(lhs₁₄ * rhs₁₄ * inv * p) |
  //            └───────────────────────────┴───────────────────────────┴─────┴─────────────────────────────┴─────────────────────────────┘
  // clang-format on
  __m512i q_p_evn = _mm512_mul_epu32(q_evn, p);
  // clang-format off
  //            ┌───────────────────────────┬───────────────────────────┬─────┬─────────────────────────────┬─────────────────────────────┐
  // |q_p_odd|: │ Lo(lhs₁ * rhs₁ * inv * p) │ Hi(lhs₁ * rhs₁ * inv * p) | ... │ Lo(lhs₁₅ * rhs₁₅ * inv * p) │ Hi(lhs₁₅ * rhs₁₅ * inv * p) |
  //            └───────────────────────────┴───────────────────────────┴─────┴─────────────────────────────┴─────────────────────────────┘
  // clang-format on
  __m512i q_p_odd = _mm512_mul_epu32(q_odd, p);

  // We can ignore all the low halves of |q_p| as they cancel out. Get all the
  // high halves as one vector.
  __m512i q_p_hi = mask_movehdup_epi32(q_p_odd, kEvens, q_p_evn);

  // Subtraction |prod_hi - q_p_hi| modulo |p|.
  // NB: Normally we'd |vpaddd P| and take the |vpminud|, but |vpminud| runs
  // on port 0, which is already under a lot of pressure performing
  // multiplications. To relieve this pressure, we check for underflow to
  // generate a mask, and then conditionally add |p|. The underflow check runs
  // on port 5, increasing our throughput, although it does cost us an
  // additional cycle of latency.
  __mmask16 underflow = _mm512_cmplt_epu32_mask(prod_hi, q_p_hi);
  __m512i t = _mm512_sub_epi32(prod_hi, q_p_hi);
  return _mm512_mask_add_epi32(t, underflow, t, p);
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX512_H_
