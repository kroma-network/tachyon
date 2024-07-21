// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

// Copyright 2022 Ethereum Foundation
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.EF and the LICENCE-APACHE.EF
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_

#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_config.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_sponge_base.h"
#include "tachyon/crypto/hashes/sponge/sponge_state.h"
#include "tachyon/math/matrix/matrix_operations.h"

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

  PoseidonSponge() = default;
  explicit PoseidonSponge(const PoseidonConfig<F>& config) : config(config) {}

  // PoseidonSpongeBase methods
  void Permute(SpongeState<F>& state) const {
    this->ApplyARKFull(state, 0);

    size_t full_rounds_over_2 = config.full_rounds / 2;
    for (size_t i = 1; i < full_rounds_over_2; ++i) {
      this->ApplySBoxFull(state);
      this->ApplyARKFull(state, i);
      ApplyMixFull(state);
    }
    this->ApplySBoxFull(state);
    this->ApplyARKFull(state, full_rounds_over_2);
    ApplyMixEfficientFull(state, full_rounds_over_2);

    for (size_t i = full_rounds_over_2 + 1;
         i < full_rounds_over_2 + config.partial_rounds + 1; ++i) {
      this->ApplySBoxPartial(state);
      this->ApplyARKPartial(state, i);
      ApplyMixEfficientPartial(state, i - (full_rounds_over_2 + 1));
    }

    for (size_t i = full_rounds_over_2 + config.partial_rounds + 1;
         i < config.partial_rounds + config.full_rounds; ++i) {
      this->ApplySBoxFull(state);
      this->ApplyARKFull(state, i);
      ApplyMixFull(state);
    }
    this->ApplySBoxFull(state);
    ApplyMixFull(state);
  }

  bool operator==(const PoseidonSponge& other) const {
    return config == other.config;
  }
  bool operator!=(const PoseidonSponge& other) const {
    return !operator==(other);
  }

 private:
  void ApplyMixFull(SpongeState<F>& state) const {
    state.elements = math::MulMatVecSerial(config.mds, state.elements);
  }

  void ApplyMixEfficientFull(SpongeState<F>& state, Eigen::Index index) const {
    state.elements =
        math::MulMatVecSerial(config.pre_sparse_mds, state.elements);
  }

  void ApplyMixEfficientPartial(SpongeState<F>& state,
                                Eigen::Index index) const {
    config.sparse_mds_matrices[index].Apply(state.elements);
  }
};

template <typename Field>
struct CryptographicSpongeTraits<PoseidonSponge<Field>> {
  using F = Field;
};

}  // namespace crypto

namespace base {

template <typename F>
class Copyable<crypto::PoseidonSponge<F>> {
 public:
  static bool WriteTo(const crypto::PoseidonSponge<F>& poseidon,
                      Buffer* buffer) {
    return buffer->WriteMany(poseidon.config);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::PoseidonSponge<F>* poseidon) {
    crypto::PoseidonConfig<F> config;
    if (!buffer.ReadMany(&config)) {
      return false;
    }

    *poseidon = crypto::PoseidonSponge<F>(std::move(config));
    return true;
  }

  static size_t EstimateSize(const crypto::PoseidonSponge<F>& poseidon) {
    return base::EstimateSize(poseidon.config);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_
