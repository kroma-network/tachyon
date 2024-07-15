// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_PACKED_BABY_BEAR_NEON_H_
#define TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_PACKED_BABY_BEAR_NEON_H_

#include <stddef.h>

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/math/finite_fields/packed_prime_field_base.h"

namespace tachyon::math {

class PackedBabyBearNeon;

template <>
struct PackedPrimeFieldTraits<PackedBabyBearNeon> {
  using PrimeField = BabyBear;

  constexpr static size_t N = 4;
};

class TACHYON_EXPORT PackedBabyBearNeon final
    : public PackedPrimeFieldBase<PackedBabyBearNeon> {
 public:
  using PrimeField = BabyBear;

  constexpr static size_t N = 4;

  PackedBabyBearNeon() = default;
  // NOTE(chokobole): This is needed by Eigen matrix.
  explicit PackedBabyBearNeon(uint32_t value);
  PackedBabyBearNeon(const PackedBabyBearNeon& other) = default;
  PackedBabyBearNeon& operator=(const PackedBabyBearNeon& other) = default;
  PackedBabyBearNeon(PackedBabyBearNeon&& other) = default;
  PackedBabyBearNeon& operator=(PackedBabyBearNeon&& other) = default;

  static void Init();

  static PackedBabyBearNeon Zero();

  static PackedBabyBearNeon One();

  static PackedBabyBearNeon MinusOne();

  static PackedBabyBearNeon Broadcast(const PrimeField& value);

  // AdditiveSemigroup methods
  PackedBabyBearNeon Add(const PackedBabyBearNeon& other) const;

  // AdditiveGroup methods
  PackedBabyBearNeon Sub(const PackedBabyBearNeon& other) const;

  PackedBabyBearNeon Negate() const;

  // MultiplicativeSemigroup methods
  PackedBabyBearNeon Mul(const PackedBabyBearNeon& other) const;
};

}  // namespace tachyon::math

#endif  //  TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_PACKED_BABY_BEAR_NEON_H_
