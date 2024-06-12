// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_CONFIG_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_CONFIG_H_

#include <array>
#include <utility>

#include "absl/types/span.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/ranges/algorithm.h"
#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_config_base.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_config_entry.h"
#include "tachyon/math/matrix/matrix_utils.h"

namespace tachyon {
namespace crypto {

// ARK(AddRoundKey) is a matrix that contains an ARC(AddRoundConstant) array in
// each row. Each constant is added to the |state| of each round of Poseidon.
// TODO(chokobole): add comment
template <typename F>
void FindPoseidon2Ark(const PoseidonGrainLFSRConfig& config,
                      math::Matrix<F>& ark) {
  PoseidonGrainLFSR<F> lfsr(config);
  ark = math::Matrix<F>(config.num_full_rounds + config.num_partial_rounds,
                        config.state_len);
  size_t partial_rounds_start = config.num_full_rounds / 2;
  size_t partial_rounds_end =
      config.num_full_rounds / 2 + config.num_partial_rounds;
  for (size_t i = 0; i < config.num_full_rounds + config.num_partial_rounds;
       ++i) {
    if (i < partial_rounds_start || i >= partial_rounds_end) {
      ark.row(i) = lfsr.GetFieldElementsRejectionSampling(config.state_len);
    } else {
      ark.row(i)[0] = lfsr.GetFieldElementsRejectionSampling(1)[0];
    }
  }
  // TODO(chokobole): Enable generating |internal_diagonal_mins_one|.
}

template <typename F>
struct Poseidon2Config : public PoseidonConfigBase<F> {
  math::Vector<F> internal_diagonal_minus_one;
  math::Vector<uint8_t> internal_shifts;

  Poseidon2Config() = default;
  Poseidon2Config(const PoseidonConfigBase<F>& base,
                  const math::Vector<F>& internal_diagonal_minus_one)
      : PoseidonConfigBase<F>(base),
        internal_diagonal_minus_one(internal_diagonal_minus_one) {}
  Poseidon2Config(PoseidonConfigBase<F>&& base,
                  math::Vector<F>&& internal_diagonal_minus_one)
      : PoseidonConfigBase<F>(std::move(base)),
        internal_diagonal_minus_one(std::move(internal_diagonal_minus_one)) {}

  template <size_t N>
  constexpr static Poseidon2Config CreateCustom(
      size_t rate, uint64_t alpha, size_t full_rounds, size_t partial_rounds,
      const std::array<F, N>& internal_diagonal_minus_one) {
    Poseidon2ConfigEntry config_entry(rate, alpha, full_rounds, partial_rounds);
    Poseidon2Config ret = config_entry.ToPoseidon2Config<F>();
    ret.internal_diagonal_minus_one = math::Vector<F>(N);
    for (size_t i = 0; i < N; ++i) {
      ret.internal_diagonal_minus_one[i] = internal_diagonal_minus_one[i];
    }
    FindPoseidon2Ark<F>(config_entry.ToPoseidonGrainLFSRConfig<F>(), ret.ark);
    return ret;
  }

  template <size_t N>
  constexpr static Poseidon2Config CreateCustom(
      size_t rate, uint64_t alpha, size_t full_rounds, size_t partial_rounds,
      const std::array<uint8_t, N>& internal_shifts) {
    Poseidon2ConfigEntry config_entry(rate, alpha, full_rounds, partial_rounds);
    Poseidon2Config ret = config_entry.ToPoseidon2Config<F>();
    ret.internal_shifts = math::Vector<uint8_t>(N);
    for (size_t i = 0; i < N; ++i) {
      ret.internal_shifts[i] = internal_shifts[i];
    }
    FindPoseidon2Ark<F>(config_entry.ToPoseidonGrainLFSRConfig<F>(), ret.ark);
    return ret;
  }
};

template <typename F>
Poseidon2Config<F> Poseidon2ConfigEntry::ToPoseidon2Config() const {
  Poseidon2Config<F> config;
  config.full_rounds = full_rounds;
  config.partial_rounds = partial_rounds;
  config.alpha = alpha;
  config.rate = rate;
  config.capacity = 1;
  return config;
}

}  // namespace crypto

namespace base {

template <typename F>
class Copyable<crypto::Poseidon2Config<F>> {
 public:
  static bool WriteTo(const crypto::Poseidon2Config<F>& config,
                      Buffer* buffer) {
    return Copyable<crypto::PoseidonConfigBase<F>>::WriteTo(config, buffer) &&
           buffer->WriteMany(config.internal_diagonal_minus_one);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::Poseidon2Config<F>* config) {
    crypto::PoseidonConfigBase<F> base;
    math::Matrix<F> internal_diagonal_minus_one;
    if (!buffer.ReadMany(&base, &internal_diagonal_minus_one)) {
      return false;
    }

    *config = {std::move(base), std::move(internal_diagonal_minus_one)};
    return true;
  }

  static size_t EstimateSize(const crypto::Poseidon2Config<F>& config) {
    const crypto::PoseidonConfigBase<F>& base =
        static_cast<const crypto::PoseidonConfigBase<F>&>(config);
    return base::EstimateSize(base, config.internal_diagonal_minus_one);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_CONFIG_H_
