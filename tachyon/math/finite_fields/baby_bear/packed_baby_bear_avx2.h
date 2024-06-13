// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_PACKED_BABY_BEAR_AVX2_H_
#define TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_PACKED_BABY_BEAR_AVX2_H_

#include <stddef.h>

#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/math/finite_fields/packed_prime_field_base.h"

namespace tachyon::math {

class PackedBabyBearAVX2;

template <>
struct PackedPrimeFieldTraits<PackedBabyBearAVX2> {
  using PrimeField = BabyBear;

  constexpr static size_t N = 8;
};

class PackedBabyBearAVX2 final
    : public PackedPrimeFieldBase<PackedBabyBearAVX2> {
 public:
  using PrimeField = BabyBear;

  constexpr static size_t N = 8;

  PackedBabyBearAVX2() = default;
  PackedBabyBearAVX2(const PackedBabyBearAVX2& other) = default;
  PackedBabyBearAVX2& operator=(const PackedBabyBearAVX2& other) = default;
  PackedBabyBearAVX2(PackedBabyBearAVX2&& other) = default;
  PackedBabyBearAVX2& operator=(PackedBabyBearAVX2&& other) = default;

  static void Init();

  static PackedBabyBearAVX2 Zero();

  static PackedBabyBearAVX2 One();

  static PackedBabyBearAVX2 Broadcast(const PrimeField& value);

  // AdditiveSemigroup methods
  PackedBabyBearAVX2 Add(const PackedBabyBearAVX2& other) const;

  // AdditiveGroup methods
  PackedBabyBearAVX2 Sub(const PackedBabyBearAVX2& other) const;

  PackedBabyBearAVX2 Negate() const;

  // MultiplicativeSemigroup methods
  PackedBabyBearAVX2 Mul(const PackedBabyBearAVX2& other) const;
};

}  // namespace tachyon::math

#endif  //  TACHYON_MATH_FINITE_FIELDS_BABY_BEAR_PACKED_BABY_BEAR_AVX2_H_
