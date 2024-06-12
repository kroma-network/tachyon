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

  SpongeState<F> state;

  Poseidon2Sponge() = default;
  explicit Poseidon2Sponge(const Poseidon2Config<F>& config)
      : config(config), state(config.rate + config.capacity) {}
  Poseidon2Sponge(const Poseidon2Config<F>& config, const SpongeState<F>& state)
      : config(config), state(state) {}
  Poseidon2Sponge(const Poseidon2Config<F>& config, SpongeState<F>&& state)
      : config(config), state(std::move(state)) {}

  // PoseidonSpongeBase methods
  void ApplyARK(Eigen::Index round_number, bool is_full_round) {
    if (is_full_round) {
      state.elements += config.ark.row(round_number);
    } else {
      state.elements[0] += config.ark.row(round_number)[0];
    }
  }

  void ApplyMix(bool is_full_round) {
    if (is_full_round) {
      ExternalMatrix::Apply(state.elements);
    } else {
      if constexpr (F::Config::kModulusBits <= 32) {
        if (config.internal_diagonal_minus_one.rows() == 0) {
          Poseidon2Plonky3InternalMatrix<F>::Apply(state.elements,
                                                   config.internal_shifts);
          return;
        }
      }
      Poseidon2HorizenInternalMatrix<F>::Apply(
          state.elements, config.internal_diagonal_minus_one);
    }
  }

  bool operator==(const Poseidon2Sponge& other) const {
    return config == other.config && state == other.state;
  }
  bool operator!=(const Poseidon2Sponge& other) const {
    return !operator==(other);
  }
};

template <typename ExternalMatrix>
struct CryptographicSpongeTraits<Poseidon2Sponge<ExternalMatrix>> {
  using F = typename ExternalMatrix::Field;
  constexpr static bool kApplyMixAtFront = true;
};

}  // namespace crypto

namespace base {

template <typename ExternalMatrix>
class Copyable<crypto::Poseidon2Sponge<ExternalMatrix>> {
 public:
  using F = typename ExternalMatrix::Field;

  static bool WriteTo(const crypto::Poseidon2Sponge<ExternalMatrix>& poseidon,
                      Buffer* buffer) {
    return buffer->WriteMany(poseidon.config, poseidon.state);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::Poseidon2Sponge<ExternalMatrix>* poseidon) {
    crypto::Poseidon2Config<F> config;
    crypto::SpongeState<F> state;
    if (!buffer.ReadMany(&config, &state)) {
      return false;
    }

    *poseidon = {std::move(config), std::move(state)};
    return true;
  }

  static size_t EstimateSize(
      const crypto::Poseidon2Sponge<ExternalMatrix>& poseidon) {
    return base::EstimateSize(poseidon.config, poseidon.state);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_H_
