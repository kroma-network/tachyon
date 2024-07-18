// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_MERSENNE31_PACKED_MERSENNE31_NEON_H_
#define TACHYON_MATH_FINITE_FIELDS_MERSENNE31_PACKED_MERSENNE31_NEON_H_

#include <stddef.h>

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/mersenne31/mersenne31.h"
#include "tachyon/math/finite_fields/packed_prime_field_base.h"

namespace tachyon::math {

class PackedMersenne31Neon;

template <>
struct PackedPrimeFieldTraits<PackedMersenne31Neon> {
  using PrimeField = Mersenne31;

  constexpr static size_t N = 4;
};

class TACHYON_EXPORT PackedMersenne31Neon final
    : public PackedPrimeFieldBase<PackedMersenne31Neon> {
 public:
  using PrimeField = Mersenne31;

  constexpr static size_t N = PackedPrimeFieldTraits<PackedMersenne31Neon>::N;

  PackedMersenne31Neon() = default;
  // NOTE(chokobole): This is needed by Eigen matrix.
  explicit PackedMersenne31Neon(uint32_t value);
  PackedMersenne31Neon(const PackedMersenne31Neon& other) = default;
  PackedMersenne31Neon& operator=(const PackedMersenne31Neon& other) = default;
  PackedMersenne31Neon(PackedMersenne31Neon&& other) = default;
  PackedMersenne31Neon& operator=(PackedMersenne31Neon&& other) = default;

  static void Init();

  static PackedMersenne31Neon Zero();

  static PackedMersenne31Neon One();

  static PackedMersenne31Neon MinusOne();

  static PackedMersenne31Neon Broadcast(const PrimeField& value);

  // AdditiveSemigroup methods
  PackedMersenne31Neon Add(const PackedMersenne31Neon& other) const;

  // AdditiveGroup methods
  PackedMersenne31Neon Sub(const PackedMersenne31Neon& other) const;

  PackedMersenne31Neon Negate() const;

  // MultiplicativeSemigroup methods
  PackedMersenne31Neon Mul(const PackedMersenne31Neon& other) const;
};

}  // namespace tachyon::math

#endif  //  TACHYON_MATH_FINITE_FIELDS_MERSENNE31_PACKED_MERSENNE31_NEON_H_
