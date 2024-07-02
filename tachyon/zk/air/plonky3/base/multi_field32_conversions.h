// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_ZK_AIR_PLONKY3_BASE_MULTI_FIELD32_CONVERSIONS_H_
#define TACHYON_ZK_AIR_PLONKY3_BASE_MULTI_FIELD32_CONVERSIONS_H_

#include <stdint.h>

#include <array>
#include <limits>

#include "absl/types/span.h"

#include "tachyon/base/containers/adapters.h"
#include "tachyon/build/build_config.h"

namespace tachyon::zk::air::plonky3 {

template <typename BigF, typename SmallF>
BigF Reduce(absl::Span<const SmallF> values) {
  static_assert(SmallF::Config::kModulusBits <= 32);
  static_assert(BigF::Config::kModulusBits > 64);

  using BigInt = typename BigF::BigIntTy;
  CHECK_LT(values.size(), BigInt::kLimbNums * 2);

  BigInt ret;
  for (size_t i = 0; i < values.size(); i += 2) {
    uint32_t value = values[i].value();
    if constexpr (SmallF::Config::kUseMontgomery) {
      ret[i >> 1] = SmallF::Config::FromMontgomery(value);
    } else {
      ret[i >> 1] = value;
    }
    if (i < values.size() - 1) {
      uint64_t value2 = values[i + 1].value();
      if constexpr (SmallF::Config::kUseMontgomery) {
        ret[i >> 1] += uint64_t{SmallF::Config::FromMontgomery(value2)} << 32;
      } else {
        ret[i >> 1] += value2 << 32;
      }
    }
  }
  return BigF(ret % BigF::Config::kModulus);
}

template <typename SmallF, typename BigF,
          size_t N = BigF::Config::kModulusBits / 64>
std::array<SmallF, N> Split(const BigF& value) {
  static_assert(SmallF::Config::kModulusBits <= 32);
  static_assert(BigF::Config::kModulusBits > 64);
  static_assert(ARCH_CPU_LITTLE_ENDIAN);

  using BigInt = typename BigF::BigIntTy;
  std::array<SmallF, N> ret;
  BigInt value_bigint = value.ToBigInt();
  for (size_t i = 0; i < N; ++i) {
    uint64_t digit = value_bigint[0] & std::numeric_limits<uint64_t>::max();
    if constexpr (SmallF::Config::kUseMontgomery) {
      ret[i] = SmallF::FromMontgomery(SmallF::Config::ToMontgomery(digit));
    } else {
      ret[i] = SmallF(digit);
    }
    value_bigint >>= 64;
  }
  return ret;
}

}  // namespace tachyon::zk::air::plonky3

#endif  // TACHYON_ZK_AIR_PLONKY3_BASE_MULTI_FIELD32_CONVERSIONS_H_
