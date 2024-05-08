// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR_AVX512_H_
#define TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR_AVX512_H_

#include <stddef.h>

#include "tachyon/math/finite_fields/koala_bear/koala_bear.h"
#include "tachyon/math/finite_fields/packed_prime_field_base.h"

namespace tachyon::math {

class PackedKoalaBearAVX512;

template <>
struct PackedPrimeFieldTraits<PackedKoalaBearAVX512> {
  using PrimeField = KoalaBear;

  constexpr static size_t N = 16;
};

class PackedKoalaBearAVX512
    : public PackedPrimeFieldBase<PackedKoalaBearAVX512> {
 public:
  using PrimeField = KoalaBear;

  constexpr static size_t N = 16;

  PackedKoalaBearAVX512() = default;
  PackedKoalaBearAVX512(const PackedKoalaBearAVX512& other) = default;
  PackedKoalaBearAVX512& operator=(const PackedKoalaBearAVX512& other) =
      default;
  PackedKoalaBearAVX512(PackedKoalaBearAVX512&& other) = default;
  PackedKoalaBearAVX512& operator=(PackedKoalaBearAVX512&& other) = default;

  static void Init();

  static PackedKoalaBearAVX512 Zero();

  static PackedKoalaBearAVX512 One();

  static PackedKoalaBearAVX512 Broadcast(const PrimeField& value);

  // AdditiveSemigroup methods
  PackedKoalaBearAVX512 Add(const PackedKoalaBearAVX512& other) const;

  // AdditiveGroup methods
  PackedKoalaBearAVX512 Sub(const PackedKoalaBearAVX512& other) const;

  PackedKoalaBearAVX512 Negate() const;

  // MultiplicativeSemigroup methods
  PackedKoalaBearAVX512 Mul(const PackedKoalaBearAVX512& other) const;
};

}  // namespace tachyon::math

#endif  //  TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR_AVX512_H_
