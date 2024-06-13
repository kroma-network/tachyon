// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR_NEON_H_
#define TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR_NEON_H_

#include <stddef.h>

#include "tachyon/math/finite_fields/koala_bear/koala_bear.h"
#include "tachyon/math/finite_fields/packed_prime_field_base.h"

namespace tachyon::math {

class PackedKoalaBearNeon;

template <>
struct PackedPrimeFieldTraits<PackedKoalaBearNeon> {
  using PrimeField = KoalaBear;

  constexpr static size_t N = 4;
};

class PackedKoalaBearNeon final
    : public PackedPrimeFieldBase<PackedKoalaBearNeon> {
 public:
  using PrimeField = KoalaBear;

  constexpr static size_t N = 4;

  PackedKoalaBearNeon() = default;
  PackedKoalaBearNeon(const PackedKoalaBearNeon& other) = default;
  PackedKoalaBearNeon& operator=(const PackedKoalaBearNeon& other) = default;
  PackedKoalaBearNeon(PackedKoalaBearNeon&& other) = default;
  PackedKoalaBearNeon& operator=(PackedKoalaBearNeon&& other) = default;

  static void Init();

  static PackedKoalaBearNeon Zero();

  static PackedKoalaBearNeon One();

  static PackedKoalaBearNeon Broadcast(const PrimeField& value);

  // AdditiveSemigroup methods
  PackedKoalaBearNeon Add(const PackedKoalaBearNeon& other) const;

  // AdditiveGroup methods
  PackedKoalaBearNeon Sub(const PackedKoalaBearNeon& other) const;

  PackedKoalaBearNeon Negate() const;

  // MultiplicativeSemigroup methods
  PackedKoalaBearNeon Mul(const PackedKoalaBearNeon& other) const;
};

}  // namespace tachyon::math

#endif  //  TACHYON_MATH_FINITE_FIELDS_KOALA_BEAR_PACKED_KOALA_BEAR_NEON_H_
