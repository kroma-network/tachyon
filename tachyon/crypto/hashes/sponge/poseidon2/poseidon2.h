// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_H_

#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_sponge_base.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_config.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_horizen_internal_matrix.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_internal_matrix.h"
#include "tachyon/crypto/hashes/sponge/sponge_state.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"

namespace tachyon {
namespace crypto {

// Poseidon2 Sponge Hash: Absorb → Permute → Squeeze
// Absorb: Absorb elements into the sponge.
// Permute: Transform the |state| using a series of operations.
//   1. Apply ARK (addition of round constants) to |state|.
//   2. Apply S-Box (xᵅ) to |state|.
//   3. Apply external and internal matrices to |state|.
// Squeeze: Squeeze elements out of the sponge.
template <typename ExternalMatrix>
struct Poseidon2Sponge final
    : public PoseidonSpongeBase<Poseidon2Sponge<ExternalMatrix>> {
  using F = typename ExternalMatrix::Field;

  // Sponge Config
  Poseidon2Config<F> config;

  Poseidon2Sponge() = default;
  explicit Poseidon2Sponge(const Poseidon2Config<F>& config) : config(config) {}
  explicit Poseidon2Sponge(Poseidon2Config<F>&& config)
      : config(std::move(config)) {}

  // PoseidonSpongeBase methods
  void Permute(SpongeState<F>& state) const {
    ApplyMixFull(state);

    size_t full_rounds_over_2 = config.full_rounds / 2;
    for (size_t i = 0; i < full_rounds_over_2; ++i) {
      this->ApplyARKFull(state, i);
      this->ApplySBoxFull(state);
      ApplyMixFull(state);
    }
    for (size_t i = full_rounds_over_2;
         i < full_rounds_over_2 + config.partial_rounds; ++i) {
      this->ApplyARKPartial(state, i);
      this->ApplySBoxPartial(state);
      ApplyMixPartial(state);
    }
    for (size_t i = full_rounds_over_2 + config.partial_rounds;
         i < config.partial_rounds + config.full_rounds; ++i) {
      this->ApplyARKFull(state, i);
      this->ApplySBoxFull(state);
      ApplyMixFull(state);
    }
  }

  bool operator==(const Poseidon2Sponge& other) const {
    return config == other.config;
  }
  bool operator!=(const Poseidon2Sponge& other) const {
    return !operator==(other);
  }

 private:
  void ApplyMixFull(SpongeState<F>& state) const {
    ExternalMatrix::Apply(state.elements);
  }

  void ApplyMixPartial(SpongeState<F>& state) const {
    using PrimeField = math::MaybeUnpack<F>;

    if constexpr (PrimeField::Config::kModulusBits <= 32) {
      if (config.use_plonky3_internal_matrix) {
        if constexpr (math::FiniteFieldTraits<F>::kIsPackedPrimeField) {
          Poseidon2Plonky3InternalMatrix<F>::Apply(
              state.elements, config.internal_diagonal_minus_one);
        } else {
          Poseidon2Plonky3InternalMatrix<F>::Apply(state.elements,
                                                   config.internal_shifts);
        }
        return;
      }
    }
    Poseidon2HorizenInternalMatrix<F>::Apply(
        state.elements, config.internal_diagonal_minus_one);
  }
};

template <typename ExternalMatrix>
struct CryptographicSpongeTraits<Poseidon2Sponge<ExternalMatrix>> {
  using F = typename ExternalMatrix::Field;
};

}  // namespace crypto

namespace base {
template <typename ExternalMatrix>
class Copyable<crypto::Poseidon2Sponge<ExternalMatrix>> {
 public:
  using F = typename ExternalMatrix::Field;

  static bool WriteTo(const crypto::Poseidon2Sponge<ExternalMatrix>& poseidon,
                      Buffer* buffer) {
    return buffer->WriteMany(poseidon.config);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::Poseidon2Sponge<ExternalMatrix>* poseidon) {
    crypto::Poseidon2Config<F> config;
    if (!buffer.ReadMany(&config)) {
      return false;
    }

    *poseidon = crypto::Poseidon2Sponge<ExternalMatrix>(std::move(config));
    return true;
  }

  static size_t EstimateSize(
      const crypto::Poseidon2Sponge<ExternalMatrix>& poseidon) {
    return base::EstimateSize(poseidon.config);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_H_
