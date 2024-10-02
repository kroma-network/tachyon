// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PLONKY3_INTERNAL_MATRIX_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PLONKY3_INTERNAL_MATRIX_H_

#include <numeric>
#include <type_traits>

#include "tachyon/base/logging.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_internal_matrix.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::crypto {

template <typename F>
class Poseidon2Plonky3InternalMatrix {
 public:
  template <size_t N, typename F2 = F,
            std::enable_if_t<
                math::FiniteFieldTraits<F2>::kIsPackedPrimeField>* = nullptr>
  static void Apply(std::array<F, N>& v,
                    const math::Vector<F>& diagonal_minus_one) {
    using PrimeField = typename math::FiniteFieldTraits<F>::PrimeField;

    Poseidon2HorizenInternalMatrix<F>::Apply(v, diagonal_minus_one);
    if constexpr (PrimeField::Config::kUseMontgomery) {
      static_assert(PrimeField::Config::kModulusBits <= 32);
      for (F& f : v) {
        f *= F::RawOne();
      }
    }
  }

  template <size_t N, typename F2 = F,
            std::enable_if_t<
                math::FiniteFieldTraits<F2>::kIsPrimeField &&
                !math::FiniteFieldTraits<F2>::kIsPackedPrimeField>* = nullptr>
  static void Apply(std::array<F, N>& v, const math::Vector<uint8_t>& shifts) {
    // |partial_sum| =       v₁ + v₂ + ... + vₙ₋₂ + vₙ₋₁
    // |full_sum|    =  v₀ + v₁ + v₂ + ... + vₙ₋₂ + vₙ₋₁
    //       s₀      = -v₀ + v₁ + v₂ + ... + vₙ₋₂ + vₙ₋₁
    //       sᵢ      = |full_sum| + (vᵢ << shiftsᵢ₋₁)
    static_assert(F::Config::kModulusBits <= 32);
    uint64_t partial_sum = std::accumulate(
        v.begin() + 1, v.end(), uint64_t{0},
        [](uint64_t acc, F value) { return acc += value.value(); });
    uint64_t full_sum = partial_sum + v[0].value();
    uint64_t s0 = partial_sum + (-v[0]).value();

    if constexpr (F::Config::kUseMontgomery) {
      v[0] = F::FromMontgomery(F::Config::FromMontgomery(s0));
    } else {
      v[0] = FromU62(s0);
    }
    for (size_t i = 1; i < N; ++i) {
      uint64_t si = full_sum + (uint64_t{v[i].value()} << shifts[i - 1]);
      if constexpr (F::Config::kUseMontgomery) {
        v[i] = F::FromMontgomery(F::Config::FromMontgomery(si));
      } else {
        v[i] = FromU62(si);
      }
    }
  }

 private:
  template <typename F2 = F,
            std::enable_if_t<
                math::FiniteFieldTraits<F2>::kIsPrimeField &&
                !math::FiniteFieldTraits<F2>::kIsPackedPrimeField>* = nullptr>
  constexpr static F FromU62(uint64_t input) {
    DCHECK_LT(input, (uint64_t{1} << 62));
    uint32_t lo = static_cast<uint32_t>(input & F::Config::kModulus);
    uint32_t hi = static_cast<uint32_t>(input >> 31);
    return F(lo) + F(hi);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_PLONKY3_INTERNAL_MATRIX_H_
