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
template <typename _Params>
struct PoseidonSponge final
    : public PoseidonSpongeBase<PoseidonSponge<_Params>> {
  using Params = _Params;
  using F = typename Params::Field;
  // Sponge Config
  PoseidonConfig<Params> config;

  PoseidonSponge() = default;
  explicit PoseidonSponge(const PoseidonConfig<Params>& config)
      : config(config) {}
  explicit PoseidonSponge(PoseidonConfig<Params>&& config)
      : config(std::move(config)) {}

  // PoseidonSpongeBase methods
  void Permute(SpongeState<Params>& state) const {
    this->ApplyARKFull(state, 0);

    size_t full_rounds_over_2 = Params::kFullRounds / 2;
    for (size_t i = 1; i < full_rounds_over_2; ++i) {
      this->ApplySBoxFull(state);
      this->ApplyARKFull(state, i);
      ApplyMixFull(state);
    }
    this->ApplySBoxFull(state);
    this->ApplyARKFull(state, full_rounds_over_2);
    ApplyMixEfficientFull(state, full_rounds_over_2);

    for (size_t i = full_rounds_over_2 + 1;
         i < full_rounds_over_2 + Params::kPartialRounds + 1; ++i) {
      this->ApplySBoxPartial(state);
      this->ApplyARKPartial(state, i);
      ApplyMixEfficientPartial(state, i - (full_rounds_over_2 + 1));
    }

    for (size_t i = full_rounds_over_2 + Params::kPartialRounds + 1;
         i < Params::kPartialRounds + Params::kFullRounds; ++i) {
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
  void ApplyMixFull(SpongeState<Params>& state) const {
    state.elements = MulMatVecSerial(config.mds, state.elements);
  }

  void ApplyMixEfficientFull(SpongeState<Params>& state,
                             Eigen::Index index) const {
    state.elements = MulMatVecSerial(config.pre_sparse_mds, state.elements);
  }

  void ApplyMixEfficientPartial(SpongeState<Params>& state,
                                Eigen::Index index) const {
    config.sparse_mds_matrices[index].Apply(state.elements);
  }

  // TODO(chokobole): This will be removed in the next commit.
  template <typename Derived, typename F, size_t N>
  static std::array<F, N> MulMatVecSerial(
      const Eigen::MatrixBase<Derived>& matrix,
      const std::array<F, N>& vector) {
    std::array<F, N> ret = {F::Zero()};
    for (size_t i = 0; i < N; ++i) {
      const auto& row = matrix.row(i);
      for (size_t j = 0; j < N; ++j) {
        ret[i] += row[j] * vector[j];
      }
    }
    return ret;
  }
};

template <typename _Params>
struct CryptographicSpongeTraits<PoseidonSponge<_Params>> {
  using Params = _Params;
  using F = typename Params::Field;
};

}  // namespace crypto

namespace base {

template <typename Params>
class Copyable<crypto::PoseidonSponge<Params>> {
 public:
  static bool WriteTo(const crypto::PoseidonSponge<Params>& poseidon,
                      Buffer* buffer) {
    return buffer->WriteMany(poseidon.config);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::PoseidonSponge<Params>* poseidon) {
    crypto::PoseidonConfig<Params> config;
    if (!buffer.ReadMany(&config)) {
      return false;
    }

    *poseidon = crypto::PoseidonSponge<Params>(std::move(config));
    return true;
  }

  static size_t EstimateSize(const crypto::PoseidonSponge<Params>& poseidon) {
    return base::EstimateSize(poseidon.config);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_H_
