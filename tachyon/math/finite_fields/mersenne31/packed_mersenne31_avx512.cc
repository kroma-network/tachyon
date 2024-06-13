// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/math/finite_fields/mersenne31/packed_mersenne31_avx512.h"

#include <immintrin.h>

#include "tachyon/math/finite_fields/packed_prime_field32_avx512.h"

namespace tachyon::math {

namespace {

__m512i kP;
__m512i kZero;
__m512i kOne;

__mmask16 kEvens = 0b0101010101010101;
__mmask16 kOdds = 0b1010101010101010;

__m512i ToVector(const PackedMersenne31AVX512& packed) {
  return _mm512_loadu_si512(packed.values().data());
}

PackedMersenne31AVX512 FromVector(__m512i vector) {
  PackedMersenne31AVX512 ret;
  _mm512_storeu_si512(ret.values().data(), vector);
  return ret;
}

__m512i Add(__m512i lhs, __m512i rhs) { return AddMod32(lhs, rhs, kP); }

__m512i Sub(__m512i lhs, __m512i rhs) { return SubMod32(lhs, rhs, kP); }

__m512i Negate(__m512i val) { return NegateMod32(val, kP); }

__m512i Mul(__m512i lhs, __m512i rhs) {
  // We want this to compile to:
  // vpaddd     lhs_evn_dbl, lhs, lhs
  // vmovshdup  rhs_odd, rhs
  // vpsrlq     lhs_odd_dbl, lhs, 31
  // vpmuludq   prod_lo_dbl, lhs_evn_dbl, rhs
  // vpmuludq   prod_odd_dbl, lhs_odd_dbl, rhs_odd
  // vmovdqa32  prod_hi, prod_odd_dbl
  // vmovshdup  prod_hi{kEvens}, prod_lo_dbl
  // vmovsldup  prod_lo_dbl{kOdds prod_odd_dbl
  // vpsrld     prod_lo, prod_lo_dbl, 1
  // vpaddd     t, prod_lo, prod_hi
  // vpsubd     u, t, p
  // vpminud    r, t, u
  // throughput: 5.5 cyc/vec (2.91 els/cyc)
  // latency: (lhs->r) 15 cyc, (rhs->r) 14 cyc

  // vpmuludq only reads the bottom 32 bits of every 64-bit quadword.
  // The even indices are already in the bottom 32 bits of a quadword, so we
  // can leave them.
  //            ┌──────┬──────┬─────┬───────┬───────┐
  // |rhs_evn|: │ rhs₀ │ rhs₁ │ ... │ rhs₁₄ │ rhs₁₅ │
  //            └──────┴──────┴─────┴───────┴───────┘
  __m512i rhs_evn = rhs;
  // Again, vpmuludq only reads the bottom 32 bits so we don't need to clear
  // the top. But we do want to double the lhs.
  //                ┌──────────┬──────────┬─────┬───────────┬───────────┐
  // |lhs_evn_dbl|: │ 2 * lhs₀ │ 2 * lhs₁ │ ... │ 2 * lhs₁₄ │ 2 * lhs₁₅ │
  //                └──────────┴──────────┴─────┴───────────┴───────────┘
  __m512i lhs_evn_dbl = _mm512_add_epi32(lhs, lhs);
  // Copy the high 32 bits in each quadword of rhs down to the low 32.
  //            ┌──────┬──────┬─────┬───────┬───────┐
  // |rhs_odd|: │ rhs₁ │ rhs₁ │ ... │ rhs₁₅ │ rhs₁₅ │
  //            └──────┴──────┴─────┴───────┴───────┘
  __m512i rhs_odd = movehdup_epi32(rhs);
  // Right shift by 31 is equivalent to moving the high 32 bits down to the
  // low 32, and then doubling it. So these are the odd indices in lhs, but
  // doubled.
  //                ┌──────────┬───┬─────┬───────────┬───┐
  // |lhs_odd_dbl|: │ 2 * lhs₁ │ 0 │ ... │ 2 * lhs₁₅ │ 0 │
  //                └──────────┴───┴─────┴───────────┴───┘
  __m512i lhs_odd_dbl = _mm512_srli_epi64(lhs, 31);

  // Multiply odd indices; since |lhs_odd_dbl| is doubled, these products are
  // also doubled.
  // clang-format off
  //                 ┌─────────────────────┬─────────────────────┬─────┬───────────────────────┬───────────────────────┐
  // |prod_odd_dbl|: │ Lo(2 * lhs₁ * rhs₁) │ Hi(2 * lhs₁ * rhs₁) | ... │ Lo(2 * lhs₁₅ * rhs₁₅) │ Hi(2 * lhs₁₅ * rhs₁₅) |
  //                 └─────────────────────┴─────────────────────┴─────┴───────────────────────┴───────────────────────┘
  // clang-format on
  __m512i prod_odd_dbl = _mm512_mul_epu32(lhs_odd_dbl, rhs_odd);
  // Multiply even indices; these are also doubled.
  // clang-format off
  //                 ┌─────────────────────┬─────────────────────┬─────┬───────────────────────┬───────────────────────┐
  // |prod_odd_dbl|: │ Lo(2 * lhs₀ * rhs₀) │ Hi(2 * lhs₀ * rhs₀) | ... │ Lo(2 * lhs₁₄ * rhs₁₄) │ Hi(2 * lhs₁₄ * rhs₁₄) |
  //                 └─────────────────────┴─────────────────────┴─────┴───────────────────────┴───────────────────────┘
  // clang-format on
  __m512i prod_evn_dbl = _mm512_mul_epu32(lhs_evn_dbl, rhs_evn);

  // TODO(chokobole): Add diagram for better explanation.
  // Move the low halves of odd products into odd positions; keep the low
  // halves of even products in even positions (where they already are). Note
  // that the products are doubled, so the result is a vector of all the low
  // halves, but doubled.
  __m512i prod_lo_dbl = mask_moveldup_epi32(prod_evn_dbl, kOdds, prod_odd_dbl);
  // Move the high halves of even products into even positions, keeping the
  // high halves of odd products where they are. The products are doubled, but
  // we are looking at (prod >> 32), which cancels out the doubling, so this
  // result is _not_ doubled.
  __m512i prod_hi = mask_movehdup_epi32(prod_odd_dbl, kEvens, prod_evn_dbl);
  // Right shift to undo the doubling.
  __m512i prod_lo = _mm512_srli_epi32(prod_lo_dbl, 1);

  // Standard addition of two 31-bit values.
  return Add(prod_lo, prod_hi);
}

}  // namespace

PackedMersenne31AVX512::PackedMersenne31AVX512(uint32_t value) {
  __m512i vector = _mm512_set1_epi32(value);
  _mm512_storeu_si512(values_.data(), vector);
}

// static
void PackedMersenne31AVX512::Init() {
  kP = _mm512_set1_epi32(Mersenne31::Config::kModulus);
  kZero = _mm512_set1_epi32(0);
  kOne = _mm512_set1_epi32(1);
}

// static
PackedMersenne31AVX512 PackedMersenne31AVX512::Zero() {
  return FromVector(kZero);
}

// static
PackedMersenne31AVX512 PackedMersenne31AVX512::One() {
  return FromVector(kOne);
}

// static
PackedMersenne31AVX512 PackedMersenne31AVX512::Broadcast(
    const PrimeField& value) {
  return FromVector(_mm512_set1_epi32(value.value()));
}

PackedMersenne31AVX512 PackedMersenne31AVX512::Add(
    const PackedMersenne31AVX512& other) const {
  return FromVector(tachyon::math::Add(ToVector(*this), ToVector(other)));
}

PackedMersenne31AVX512 PackedMersenne31AVX512::Sub(
    const PackedMersenne31AVX512& other) const {
  return FromVector(tachyon::math::Sub(ToVector(*this), ToVector(other)));
}

PackedMersenne31AVX512 PackedMersenne31AVX512::Negate() const {
  return FromVector(tachyon::math::Negate(ToVector(*this)));
}

PackedMersenne31AVX512 PackedMersenne31AVX512::Mul(
    const PackedMersenne31AVX512& other) const {
  return FromVector(tachyon::math::Mul(ToVector(*this), ToVector(other)));
}

}  // namespace tachyon::math
