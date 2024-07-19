// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_MERSENNE31_PACKED_MERSENNE31_AVX2_H_
#define TACHYON_MATH_FINITE_FIELDS_MERSENNE31_PACKED_MERSENNE31_AVX2_H_

#include <stddef.h>

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/mersenne31/mersenne31.h"
#include "tachyon/math/finite_fields/packed_prime_field_base.h"

namespace tachyon::math {

class PackedMersenne31AVX2;

template <>
struct PackedFieldTraits<PackedMersenne31AVX2> {
  using PrimeField = Mersenne31;

  constexpr static size_t N = 8;
};

class TACHYON_EXPORT PackedMersenne31AVX2 final
    : public PackedPrimeFieldBase<PackedMersenne31AVX2> {
 public:
  using PrimeField = Mersenne31;

  constexpr static size_t N = PackedFieldTraits<PackedMersenne31AVX2>::N;

  PackedMersenne31AVX2() = default;
  // NOTE(chokobole): This is needed by Eigen matrix.
  explicit PackedMersenne31AVX2(uint32_t value);
  PackedMersenne31AVX2(const PackedMersenne31AVX2& other) = default;
  PackedMersenne31AVX2& operator=(const PackedMersenne31AVX2& other) = default;
  PackedMersenne31AVX2(PackedMersenne31AVX2&& other) = default;
  PackedMersenne31AVX2& operator=(PackedMersenne31AVX2&& other) = default;

  static void Init();

  static PackedMersenne31AVX2 Zero();

  static PackedMersenne31AVX2 One();

  static PackedMersenne31AVX2 MinusOne();

  static PackedMersenne31AVX2 Broadcast(const PrimeField& value);

  // AdditiveSemigroup methods
  PackedMersenne31AVX2 Add(const PackedMersenne31AVX2& other) const;

  // AdditiveGroup methods
  PackedMersenne31AVX2 Sub(const PackedMersenne31AVX2& other) const;

  PackedMersenne31AVX2 Negate() const;

  // MultiplicativeSemigroup methods
  PackedMersenne31AVX2 Mul(const PackedMersenne31AVX2& other) const;
};

}  // namespace tachyon::math

#endif  //  TACHYON_MATH_FINITE_FIELDS_MERSENNE31_PACKED_MERSENNE31_AVX2_H_
