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
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_state.h"

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
template <typename PrimeField>
struct PoseidonSponge final
    : public PoseidonSpongeBase<PoseidonSponge<PrimeField>> {
  using F = PrimeField;

  // Sponge Config
  PoseidonConfig<F> config;

  // Sponge State
  PoseidonState<F> state;

  PoseidonSponge() = default;
  explicit PoseidonSponge(const PoseidonConfig<F>& config)
      : config(config), state(config.rate + config.capacity) {}
  PoseidonSponge(const PoseidonConfig<F>& config, const PoseidonState<F>& state)
      : config(config), state(state) {}
  PoseidonSponge(const PoseidonConfig<F>& config, PoseidonState<F>&& state)
      : config(config), state(std::move(state)) {}

  // PoseidonSpongeBase methods
  void ApplyMix(bool) { state.elements = config.mds * state.elements; }

  bool operator==(const PoseidonSponge& other) const {
    return config == other.config && state == other.state;
  }
  bool operator!=(const PoseidonSponge& other) const {
    return !operator==(other);
  }
};

template <typename PrimeField>
struct CryptographicSpongeTraits<PoseidonSponge<PrimeField>> {
  using F = PrimeField;
  constexpr static bool kApplyMixAtFront = false;
};

}  // namespace crypto

namespace base {

template <typename PrimeField>
class Copyable<crypto::PoseidonSponge<PrimeField>> {
 public:
  static bool WriteTo(const crypto::PoseidonSponge<PrimeField>& poseidon,
                      Buffer* buffer) {
    return buffer->WriteMany(poseidon.config, poseidon.state);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::PoseidonSponge<PrimeField>* poseidon) {
    crypto::PoseidonConfig<PrimeField> config;
    crypto::PoseidonState<PrimeField> state;
    if (!buffer.ReadMany(&config, &state)) {
      return false;
    }

    *poseidon = {std::move(config), std::move(state)};
    return true;
  }

  static size_t EstimateSize(
      const crypto::PoseidonSponge<PrimeField>& poseidon) {
    return base::EstimateSize(poseidon.config, poseidon.state);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_
