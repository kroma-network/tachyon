// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_INTERNAL_PACKED_KOALA_BEAR_AVX2_H_
#define TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_INTERNAL_PACKED_KOALA_BEAR_AVX2_H_

#include <stddef.h>

#include "tachyon/export.h"
#include "tachyon/math/finite_fields/koala_bear/internal/koala_bear.h"
#include "tachyon/math/finite_fields/packed_prime_field_base.h"

namespace tachyon::math {

class PackedKoalaBearAVX2;

template <>
struct PackedFieldTraits<PackedKoalaBearAVX2> {
  using Field = KoalaBear;

  constexpr static size_t N = 8;
};

class TACHYON_EXPORT PackedKoalaBearAVX2 final
    : public PackedPrimeFieldBase<PackedKoalaBearAVX2> {
 public:
  using PrimeField = KoalaBear;

  constexpr static size_t N = 8;

  PackedKoalaBearAVX2() = default;
  // NOTE(chokobole): This is needed by Eigen matrix.
  explicit PackedKoalaBearAVX2(uint32_t value);
  PackedKoalaBearAVX2(const PackedKoalaBearAVX2& other) = default;
  PackedKoalaBearAVX2& operator=(const PackedKoalaBearAVX2& other) = default;
  PackedKoalaBearAVX2(PackedKoalaBearAVX2&& other) = default;
  PackedKoalaBearAVX2& operator=(PackedKoalaBearAVX2&& other) = default;

  static void Init();

  static PackedKoalaBearAVX2 Zero();

  static PackedKoalaBearAVX2 One();

  static PackedKoalaBearAVX2 MinusOne();

  static PackedKoalaBearAVX2 TwoInv();

  static PackedKoalaBearAVX2 Broadcast(const PrimeField& value);

  // AdditiveSemigroup methods
  PackedKoalaBearAVX2 Add(const PackedKoalaBearAVX2& other) const;

  // AdditiveGroup methods
  PackedKoalaBearAVX2 Sub(const PackedKoalaBearAVX2& other) const;

  PackedKoalaBearAVX2 Negate() const;

  // MultiplicativeSemigroup methods
  PackedKoalaBearAVX2 Mul(const PackedKoalaBearAVX2& other) const;
};

}  // namespace tachyon::math

#endif  //  TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_INTERNAL_PACKED_KOALA_BEAR_AVX2_H_
