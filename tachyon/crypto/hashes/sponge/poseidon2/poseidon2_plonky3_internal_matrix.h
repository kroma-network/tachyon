// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PLONKY3_INTERNAL_MATRIX_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PLONKY3_INTERNAL_MATRIX_H_

#include <numeric>

#include "tachyon/base/logging.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::crypto {

template <typename PrimeField>
class Poseidon2Plonky3InternalMatrix {
 public:
  static void Apply(math::Vector<PrimeField>& v,
                    const math::Vector<uint8_t>& shifts) {
    // |partial_sum| =       v₁ + v₂ + ... + vₙ₋₂ + vₙ₋₁
    // |full_sum|    =  v₀ + v₁ + v₂ + ... + vₙ₋₂ + vₙ₋₁
    //       s₀      = -v₀ + v₁ + v₂ + ... + vₙ₋₂ + vₙ₋₁
    //       sᵢ      = |full_sum| + (vᵢ << shiftsᵢ₋₁)
    static_assert(PrimeField::Config::kModulusBits <= 32);
    uint64_t partial_sum = std::accumulate(
        v.begin() + 1, v.end(), uint64_t{0},
        [](uint64_t acc, PrimeField value) { return acc += value.value(); });
    uint64_t full_sum = partial_sum + v[0].value();
    uint64_t s0 = partial_sum + (-v[0]).value();

    if constexpr (PrimeField::Config::kUseMontgomery) {
      v[0] = PrimeField::FromMontgomery(PrimeField::Config::FromMontgomery(s0));
    } else {
      v[0] = FromU62(s0);
    }
    for (Eigen::Index i = 1; i < v.size(); ++i) {
      uint64_t si = full_sum + (uint64_t{v[i].value()} << shifts[i - 1]);
      if constexpr (PrimeField::Config::kUseMontgomery) {
        v[i] =
            PrimeField::FromMontgomery(PrimeField::Config::FromMontgomery(si));
      } else {
        v[i] = FromU62(si);
      }
    }
  }

 private:
  constexpr static PrimeField FromU62(uint64_t input) {
    DCHECK_LT(input, (uint64_t{1} << 62));
    uint32_t lo = static_cast<uint32_t>(input & PrimeField::Config::kModulus);
    uint32_t hi = static_cast<uint32_t>(input >> 31);
    return PrimeField(lo) + PrimeField(hi);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PLONKY3_INTERNAL_MATRIX_H_
