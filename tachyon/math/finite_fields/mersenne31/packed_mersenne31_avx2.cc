// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/math/finite_fields/mersenne31/packed_mersenne31_avx2.h"

#include <immintrin.h>

#include "tachyon/math/finite_fields/packed_prime_field32_avx2.h"

namespace tachyon::math {

namespace {

__m256i kP;
__m256i kZero;
__m256i kOne;

__m256i ToVector(const PackedMersenne31AVX2& packed) {
  return _mm256_loadu_si256(
      reinterpret_cast<const __m256i_u*>(packed.values().data()));
}

PackedMersenne31AVX2 FromVector(__m256i vector) {
  PackedMersenne31AVX2 ret;
  _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(ret.values().data()),
                      vector);
  return ret;
}

__m256i Add(__m256i lhs, __m256i rhs) { return AddMod32(lhs, rhs, kP); }

__m256i Sub(__m256i lhs, __m256i rhs) { return SubMod32(lhs, rhs, kP); }

__m256i Neg(__m256i val) { return NegMod32(val, kP); }

__m256i movehdup_epi32(__m256i x) {
  // The instruction is only available in the floating-point flavor; this
  // distinction is only for historical reasons and no longer matters. We cast
  // to floats, duplicate, and cast back.
  return _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)));
}

__m256i Mul(__m256i lhs, __m256i rhs) {
  // We want this to compile to:
  // vpsrlq     lhs_odd_dbl, lhs, 31
  // vmovshdup  rhs_odd, rhs
  // vpmuludq   prod_odd_dbl, lhs_odd_dbl, rhs_odd
  // vpmuludq   prod_evn, lhs, rhs
  // vpsllq     prod_odd_lo_dirty, prod_odd_dbl, 31
  // vpsrlq     prod_evn_hi, prod_evn, 31
  // vpblendd   prod_lo_dirty, prod_evn, prod_odd_lo_dirty, aah
  // vpblendd   prod_hi, prod_evn_hi, prod_odd_dbl, aah
  // vpand      prod_lo, prod_lo_dirty, P
  // vpaddd     t, prod_lo, prod_hi
  // vpsubd     u, t, P
  // vpminud    r, t, u
  // throughput: 4 cyc/vec (2 els/cyc)
  // latency: 13 cyc

  // vpmuludq only reads the bottom 32 bits of every 64-bit quadword.
  // The even indices are already in the bottom 32 bits of a quadword, so we
  // can leave them.
  __m256i lhs_evn = lhs;
  __m256i rhs_evn = rhs;
  // Right shift by 31 is equivalent to moving the high 32 bits down to the
  // low 32, and then doubling it. So these are the odd indices in lhs, but
  // doubled.
  __m256i lhs_odd_dbl = _mm256_srli_epi64(lhs, 31);
  // Copy the high 32 bits in each quadword of rhs down to the low 32.
  __m256i rhs_odd = movehdup_epi32(rhs);

  // Multiply odd indices; since lhs_odd_dbl is doubled, these products are
  // also doubled. prod_odd_dbl.quadword[i] = 2 * lhs.doubleword[2 * i + 1] *
  // rhs.doubleword[2 * i + 1]
  __m256i prod_odd_dbl = _mm256_mul_epu32(rhs_odd, lhs_odd_dbl);
  // Multiply even indices.
  // prod_evn.quadword[i] = lhs.doubleword[2 * i] * rhs.doubleword[2 * i]
  __m256i prod_evn = _mm256_mul_epu32(rhs_evn, lhs_evn);

  // We now need to extract the low 31 bits and the high 31 bits of each 62
  // bit product and prepare to add them. Put the low 31 bits of the product
  // (recall that it is shifted left by 1) in an odd doubleword. (Notice that
  // the high 31 bits are already in an odd doubleword in prod_odd_dbl.) We
  // will still need to clear the sign bit, hence we mark it _dirty.
  __m256i prod_odd_lo_dirty = _mm256_slli_epi64(prod_odd_dbl, 31);
  // Put the high 31 bits in an even doubleword, again noting that in prod_evn
  // the even doublewords contain the low 31 bits (with a dirty sign bit).
  __m256i prod_evn_hi = _mm256_srli_epi64(prod_evn, 31);

  // Put all the low halves of all the products into one vector. Take the even
  // values from prod_evn and odd values from prod_odd_lo_dirty. Note that the
  // sign bits still need clearing.
  __m256i prod_lo_dirty =
      _mm256_blend_epi32(prod_evn, prod_odd_lo_dirty, 0b10101010);
  // Now put all the high halves into one vector. The even values come from
  // prod_evn_hi and the odd values come from prod_odd_dbl.
  __m256i prod_hi = _mm256_blend_epi32(prod_evn_hi, prod_odd_dbl, 0b10101010);
  // Clear the most significant bit.
  __m256i prod_lo = _mm256_and_si256(prod_lo_dirty, kP);

  // Standard addition of two 31-bit values.
  return Add(prod_lo, prod_hi);
}

}  // namespace

// static
void PackedMersenne31AVX2::Init() {
  kP = _mm256_set1_epi32(Mersenne31::Config::kModulus[0]);
  kZero = _mm256_set1_epi32(0);
  kOne = _mm256_set1_epi32(1);
}

// static
PackedMersenne31AVX2 PackedMersenne31AVX2::Zero() { return FromVector(kZero); }

// static
PackedMersenne31AVX2 PackedMersenne31AVX2::One() { return FromVector(kOne); }

// static
PackedMersenne31AVX2 PackedMersenne31AVX2::Broadcast(const PrimeField& value) {
  return FromVector(_mm256_set1_epi32(value.value()));
}

PackedMersenne31AVX2 PackedMersenne31AVX2::Add(
    const PackedMersenne31AVX2& other) const {
  return FromVector(tachyon::math::Add(ToVector(*this), ToVector(other)));
}

PackedMersenne31AVX2 PackedMersenne31AVX2::Sub(
    const PackedMersenne31AVX2& other) const {
  return FromVector(tachyon::math::Sub(ToVector(*this), ToVector(other)));
}

PackedMersenne31AVX2 PackedMersenne31AVX2::Negate() const {
  return FromVector(tachyon::math::Neg(ToVector(*this)));
}

PackedMersenne31AVX2 PackedMersenne31AVX2::Mul(
    const PackedMersenne31AVX2& other) const {
  return FromVector(tachyon::math::Mul(ToVector(*this), ToVector(other)));
}

}  // namespace tachyon::math
