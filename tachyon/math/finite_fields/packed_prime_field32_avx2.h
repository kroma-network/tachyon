// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX2_H_
#define TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX2_H_

#include <immintrin.h>

#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/functional/callback.h"

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

// We provide 2 variants of Montgomery reduction depending on if the inputs are
// unsigned or signed. The unsigned variant follows steps 1 and 2 in the above
// protocol to produce D in (-P, ..., P). For the signed variant we assume -PB/2
// < C < PB/2 and let Q := μ C mod B be the unique representative in [-B/2, ...,
// B/2 - 1]. The division in step 2 is clearly still exact and |C - Q P| <= |C|
// + |Q||P| < PB so D still lies in (-P, ..., P).

// Perform a partial Montgomery reduction on each 64 bit element.
// Input must lie in {0, ..., 2³²P}.
// The output will lie in {-P, ..., P} and be stored in the upper 32 bits.
ALWAYS_INLINE __m256i PartialMontyRedUnsignedToSigned(__m256i input, __m256i p,
                                                      __m256i inv) {
  __m256i q = _mm256_mul_epu32(input, inv);
  __m256i q_p = _mm256_mul_epu32(q, p);
  // By construction, the bottom 32 bits of input and q_p are equal.
  // Thus |_mm256_sub_epi32| and |_mm256_sub_epi64| should act identically.
  // However for some reason, the compiler gets confused if we use
  // |_mm256_sub_epi64| and outputs a load of nonsense, see:
  // https://godbolt.org/z/3W8M7Tv84.
  return _mm256_sub_epi32(input, q_p);
}
// Perform a partial Montgomery reduction on each 64 bit element.
// Input must lie in {-2³¹P, ..., 2³¹P}.
// The output will lie in {-P, ..., P} and be stored in the upper 32 bits.
ALWAYS_INLINE __m256i PartialMontyRedSignedToSigned(__m256i input, __m256i p,
                                                    __m256i inv) {
  __m256i q = _mm256_mul_epi32(input, inv);
  __m256i q_p = _mm256_mul_epi32(q, p);
  // Unlike the previous case the compiler output is essentially identical
  // between |_mm256_sub_epi32| and |_mm256_sub_epi64|. We use
  // |_mm256_sub_epi32| again just for consistency.
  return _mm256_sub_epi32(input, q_p);
}

// Multiply the field elements in the even index entries.
// |lhs[2i]|, |rhs[2i]| must be unsigned 32-bit integers such that
// |lhs[2i]| * |rhs[2i]| lies in {0, ..., 2³²P}.
// The output will lie in {-P, ..., P} and be stored in |output[2i + 1]|.
ALWAYS_INLINE __m256i MontyMul(__m256i lhs, __m256i rhs, __m256i p,
                               __m256i inv) {
  __m256i prod = _mm256_mul_epu32(lhs, rhs);
  return PartialMontyRedSignedToSigned(prod, p, inv);
}

// Multiply the field elements in the even index entries.
// |lhs[2i]|, |rhs[2i]| must be signed 32-bit integers such that
// |lhs[2i]| * |rhs[2i]| lies in {-2³¹P, ..., 2³¹P}.
// The output will lie in {-P, ..., P} stored in |output[2i + 1]|.
ALWAYS_INLINE __m256i MontyMulSigned(__m256i lhs, __m256i rhs, __m256i p,
                                     __m256i inv) {
  __m256i prod = _mm256_mul_epi32(lhs, rhs);
  return PartialMontyRedSignedToSigned(prod, p, inv);
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

  __m256i d_evn = MontyMul(lhs_evn, rhs_evn, p, inv);
  __m256i d_odd = MontyMul(lhs_odd, rhs_odd, p, inv);

  __m256i d_evn_hi = movehdup_epi32(d_evn);
  __m256i t = _mm256_blend_epi32(d_evn_hi, d_odd, 0b10101010);

  __m256i u = _mm256_add_epi32(t, p);
  return _mm256_min_epu32(t, u);
}

// Square the field elements in the even index entries.
// Inputs must be signed 32-bit integers.
// Outputs will be a signed integer in (-P, ..., P) copied into both the even
// and odd indices.
ALWAYS_INLINE __m256i ShiftedSquare(__m256i input, __m256i p, __m256i inv) {
  // Note that we do not need a restriction on the size of |input[i]²| as
  // 2³⁰ < P and |i32| <= 2³¹ and so => |input[i]²| <= 2⁶² < 2³²P.
  __m256i square = _mm256_mul_epi32(input, input);
  __m256i square_red = PartialMontyRedSignedToSigned(square, p, inv);
  return movehdup_epi32(square_red);
}

// Apply callback to the even and odd indices of the input vector.
// callback should only depend in the 32 bit entries in the even indices.
// The output of callback must lie in (-P, ..., P) and be stored in the odd
// indices. The even indices of the output of callback will not be read. The
// input should conform to the requirements of |callback|.
// NOTE(chokobole): This is to suppress the error below.
// clang-format off
// error: ignoring attributes on template argument '__m256i(__m256i, __m256i, __m256i)' [-Werror=ignored-attributes]
// clang-format on
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
ALWAYS_INLINE __m256i ApplyFuncToEvenOdd(
    __m256i input, __m256i p, __m256i inv,
    base::RepeatingCallback<__m256i(__m256i, __m256i, __m256i)> callback) {
  __m256i input_evn = input;
  __m256i input_odd = movehdup_epi32(input);
  __m256i d_evn = callback.Run(input_evn, p, inv);
  __m256i d_odd = callback.Run(input_odd, p, inv);
  __m256i d_evn_hi = movehdup_epi32(d_evn);
  __m256i t = _mm256_blend_epi32(d_evn_hi, d_odd, 0b10101010);
  __m256i u = _mm256_add_epi32(t, p);
  return _mm256_min_epu32(t, u);
}
#pragma GCC diagnostic pop

// Cube the field elements in the even index entries.
// Inputs must be signed 32-bit integers in [-P, ..., P].
// Outputs will be a signed integer in (-P, ..., P) stored in the odd indices.
ALWAYS_INLINE __m256i DoExp3(__m256i input, __m256i p, __m256i inv) {
  __m256i square = ShiftedSquare(input, p, inv);
  return MontyMulSigned(square, input, p, inv);
}

ALWAYS_INLINE __m256i Exp3(__m256i input, __m256i p, __m256i inv) {
  return ApplyFuncToEvenOdd(input, p, inv, &DoExp3);
}

// Take the fifth power of the field elements in the even index
// entries. Inputs must be signed 32-bit integers in [-P, ..., P]. Outputs will
// be a signed integer in (-P, ..., P) stored in the odd indices.
ALWAYS_INLINE __m256i DoExp5(__m256i input, __m256i p, __m256i inv) {
  __m256i square = ShiftedSquare(input, p, inv);
  __m256i quad = ShiftedSquare(square, p, inv);
  return MontyMulSigned(quad, input, p, inv);
}

ALWAYS_INLINE __m256i Exp5(__m256i input, __m256i p, __m256i inv) {
  return ApplyFuncToEvenOdd(input, p, inv, &DoExp5);
}

/// Take the seventh power of the field elements in the even index
/// entries. Inputs must lie in [-P, ..., P]. Outputs will also lie in (-P, ...,
/// P) stored in the odd indices.
ALWAYS_INLINE __m256i DoExp7(__m256i input, __m256i p, __m256i inv) {
  __m256i square = ShiftedSquare(input, p, inv);
  __m256i cube = MontyMulSigned(square, input, p, inv);
  __m256i cube_shifted = movehdup_epi32(cube);
  __m256i quad = ShiftedSquare(square, p, inv);
  return MontyMulSigned(quad, cube_shifted, p, inv);
}

ALWAYS_INLINE __m256i Exp7(__m256i input, __m256i p, __m256i inv) {
  return ApplyFuncToEvenOdd(input, p, inv, &DoExp7);
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_PACKED_PRIME_FIELD32_AVX2_H_
