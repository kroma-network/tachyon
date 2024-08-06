// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/math/finite_fields/baby_bear/internal/packed_baby_bear_avx2.h"

#include <immintrin.h>

#include "tachyon/math/finite_fields/packed_prime_field32_avx2.h"

namespace tachyon::math {

namespace {

__m256i kP;
__m256i kInv;
__m256i kZero;
__m256i kOne;
__m256i kMinusOne;

__m256i ToVector(const PackedBabyBearAVX2& packed) {
  return _mm256_loadu_si256(
      reinterpret_cast<const __m256i_u*>(packed.values().data()));
}

PackedBabyBearAVX2 FromVector(__m256i vector) {
  PackedBabyBearAVX2 ret;
  _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(ret.values().data()),
                      vector);
  return ret;
}

__m256i Add(__m256i lhs, __m256i rhs) { return AddMod32(lhs, rhs, kP); }

__m256i Sub(__m256i lhs, __m256i rhs) { return SubMod32(lhs, rhs, kP); }

__m256i Negate(__m256i val) { return NegateMod32(val, kP); }

__m256i Mul(__m256i lhs, __m256i rhs) {
  return MontMulMod32(lhs, rhs, kP, kInv);
}

}  // namespace

PackedBabyBearAVX2::PackedBabyBearAVX2(uint32_t value) {
  __m256i vector = _mm256_set1_epi32(BabyBear::Config::ToMontgomery(value));
  _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(values_.data()), vector);
}

// static
void PackedBabyBearAVX2::Init() {
  BabyBear::Init();
  kP = _mm256_set1_epi32(BabyBear::Config::kModulus);
  kInv = _mm256_set1_epi32(BabyBear::Config::kInverse32);
  kZero = _mm256_set1_epi32(0);
  kOne = _mm256_set1_epi32(BabyBear::Config::kOne);
  kMinusOne = _mm256_set1_epi32(BabyBear::Config::kMinusOne);
}

// static
PackedBabyBearAVX2 PackedBabyBearAVX2::Zero() { return FromVector(kZero); }

// static
PackedBabyBearAVX2 PackedBabyBearAVX2::One() { return FromVector(kOne); }

// static
PackedBabyBearAVX2 PackedBabyBearAVX2::MinusOne() {
  return FromVector(kMinusOne);
}

// static
PackedBabyBearAVX2 PackedBabyBearAVX2::Broadcast(const PrimeField& value) {
  return FromVector(_mm256_set1_epi32(value.value()));
}

PackedBabyBearAVX2 PackedBabyBearAVX2::Add(
    const PackedBabyBearAVX2& other) const {
  return FromVector(math::Add(ToVector(*this), ToVector(other)));
}

PackedBabyBearAVX2 PackedBabyBearAVX2::Sub(
    const PackedBabyBearAVX2& other) const {
  return FromVector(math::Sub(ToVector(*this), ToVector(other)));
}

PackedBabyBearAVX2 PackedBabyBearAVX2::Negate() const {
  return FromVector(math::Negate(ToVector(*this)));
}

PackedBabyBearAVX2 PackedBabyBearAVX2::Mul(
    const PackedBabyBearAVX2& other) const {
  return FromVector(math::Mul(ToVector(*this), ToVector(other)));
}

}  // namespace tachyon::math
