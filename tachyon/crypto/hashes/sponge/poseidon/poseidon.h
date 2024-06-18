// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_

#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_config.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_sponge_base.h"
#include "tachyon/crypto/hashes/sponge/sponge_state.h"

namespace tachyon {
namespace crypto {

// Poseidon Sponge Hash: Absorb → Permute → Squeeze
// Absorb: Absorb elements into the sponge.
// Permute: Transform the |state| using a series of operations.
//   1. Apply ARK (addition of round constants) to |state|.
//   2. Apply S-Box (xᵅ) to |state|.
//   3. Apply MDS matrix to |state|.
// Squeeze: Squeeze elements out of the sponge.
// This implementation of Poseidon is entirely Fractal's implementation in
// [COS20][cos] with small syntax changes. See https://eprint.iacr.org/2019/1076
template <typename F>
struct PoseidonSponge final : public PoseidonSpongeBase<PoseidonSponge<F>> {
  // Sponge Config
  PoseidonConfig<F> config;

  SpongeState<F> state;

  PoseidonSponge() = default;
  explicit PoseidonSponge(const PoseidonConfig<F>& config)
      : config(config), state(config.rate + config.capacity) {}
  PoseidonSponge(const PoseidonConfig<F>& config, const SpongeState<F>& state)
      : config(config), state(state) {}
  PoseidonSponge(const PoseidonConfig<F>& config, SpongeState<F>&& state)
      : config(config), state(std::move(state)) {}

  // PoseidonSpongeBase methods
  void ApplyARK(Eigen::Index round_number, bool) {
    state.elements += config.ark.row(round_number);
  }

  void ApplyMix(bool) {
    // NOTE (chokobole): Eigen matrix multiplication has a computational
    // overhead unlike naive matrix multiplication.
    //
    // Example: Multiplying a 2x2 matrix by a 2x1 vector:
    //
    // +----+----+   +----+
    // | m₀ | m₁ | * | v₀ |
    // +----+----+   +----+
    // | m₂ | m₃ |   | v₁ |
    // +----+----+   +----+
    //
    // The operations involved in this multiplication are as follows:
    //
    // 1 * 1
    // 1 * 1
    // m₀ * v₀
    // m₀v₀ + 0
    // m₂ * v₀
    // m₂v₀ + 0
    // m₁ * v₁
    // m₁v₁ + m₀v₀
    // m₃ * v₁
    // m₃v₁ + m₂v₀
    // m₁v₁ + m₀v₀ * 1
    // m₁v₁ + m₀v₀ + 0
    // m₃v₁ + m₂v₀ * 1
    // m₃v₁ + m₂v₀ + 0
    math::Vector<F> elements(state.elements.size());
    for (Eigen::Index i = 0; i < config.mds.rows(); ++i) {
      for (Eigen::Index j = 0; j < config.mds.cols(); ++j) {
        elements[i] += config.mds(i, j) * state.elements[j];
      }
    }
    state.elements = std::move(elements);
  }

  bool operator==(const PoseidonSponge& other) const {
    return config == other.config && state == other.state;
  }
  bool operator!=(const PoseidonSponge& other) const {
    return !operator==(other);
  }
};

template <typename Field>
struct CryptographicSpongeTraits<PoseidonSponge<Field>> {
  using F = Field;
  constexpr static bool kApplyMixAtFront = false;
};

}  // namespace crypto

namespace base {

template <typename F>
class Copyable<crypto::PoseidonSponge<F>> {
 public:
  static bool WriteTo(const crypto::PoseidonSponge<F>& poseidon,
                      Buffer* buffer) {
    return buffer->WriteMany(poseidon.config, poseidon.state);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::PoseidonSponge<F>* poseidon) {
    crypto::PoseidonConfig<F> config;
    crypto::SpongeState<F> state;
    if (!buffer.ReadMany(&config, &state)) {
      return false;
    }

    *poseidon = {std::move(config), std::move(state)};
    return true;
  }

  static size_t EstimateSize(const crypto::PoseidonSponge<F>& poseidon) {
    return base::EstimateSize(poseidon.config, poseidon.state);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_
