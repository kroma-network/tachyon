// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_MERSENNE31_INTERNAL_PACKED_MERSENNE31_AVX512_H_
#define TACHYON_MATH_FINITE_FIELDS_MERSENNE31_INTERNAL_PACKED_MERSENNE31_AVX512_H_

#include <stddef.h>

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/mersenne31/internal/mersenne31.h"
#include "tachyon/math/finite_fields/packed_prime_field_base.h"

namespace tachyon::math {

class PackedMersenne31AVX512;

template <>
struct PackedFieldTraits<PackedMersenne31AVX512> {
  using Field = Mersenne31;

  constexpr static size_t N = 16;
};

class TACHYON_EXPORT PackedMersenne31AVX512 final
    : public PackedPrimeFieldBase<PackedMersenne31AVX512> {
 public:
  using PrimeField = Mersenne31;

  constexpr static size_t N = PackedFieldTraits<PackedMersenne31AVX512>::N;

  PackedMersenne31AVX512() = default;
  // NOTE(chokobole): This is needed by Eigen matrix.
  explicit PackedMersenne31AVX512(uint32_t value);
  PackedMersenne31AVX512(const PackedMersenne31AVX512& other) = default;
  PackedMersenne31AVX512& operator=(const PackedMersenne31AVX512& other) =
      default;
  PackedMersenne31AVX512(PackedMersenne31AVX512&& other) = default;
  PackedMersenne31AVX512& operator=(PackedMersenne31AVX512&& other) = default;

  static void Init();

  static PackedMersenne31AVX512 Zero();

  static PackedMersenne31AVX512 One();

  static PackedMersenne31AVX512 MinusOne();

  static PackedMersenne31AVX512 TwoInv();

  static PackedMersenne31AVX512 Broadcast(const PrimeField& value);

  // AdditiveSemigroup methods
  PackedMersenne31AVX512 Add(const PackedMersenne31AVX512& other) const;

  // AdditiveGroup methods
  PackedMersenne31AVX512 Sub(const PackedMersenne31AVX512& other) const;

  PackedMersenne31AVX512 Negate() const;

  // MultiplicativeSemigroup methods
  PackedMersenne31AVX512 Mul(const PackedMersenne31AVX512& other) const;
};

}  // namespace tachyon::math

#endif  //  TACHYON_MATH_FINITE_FIELDS_MERSENNE31_INTERNAL_PACKED_MERSENNE31_AVX512_H_
