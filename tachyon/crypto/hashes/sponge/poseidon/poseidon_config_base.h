// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_BASE_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_BASE_H_

#include <stddef.h>
#include <stdint.h>

#include <utility>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/crypto/hashes/sponge/sponge_config.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon {
namespace crypto {

template <typename F>
struct PoseidonConfigBase : public SpongeConfig {
  // Number of rounds in a full-round operation.
  size_t full_rounds = 0;

  // Number of rounds in a partial-round operation.
  size_t partial_rounds = 0;

  // Exponent used in S-boxes.
  uint64_t alpha = 0;

  // Additive Round Keys added before each MDS matrix application to make it an
  // affine shift. They are indexed by |ark[round_num][state_element_index]|.
  math::Matrix<F> ark;

  PoseidonConfigBase() = default;
  PoseidonConfigBase(size_t full_rounds, size_t partial_rounds, uint64_t alpha,
                     const math::Matrix<F>& ark, size_t rate, size_t capacity)
      : SpongeConfig(rate, capacity),
        full_rounds(full_rounds),
        partial_rounds(partial_rounds),
        alpha(alpha),
        ark(ark) {}
  PoseidonConfigBase(size_t full_rounds, size_t partial_rounds, uint64_t alpha,
                     math::Matrix<F>&& ark, size_t rate, size_t capacity)
      : SpongeConfig(rate, capacity),
        full_rounds(full_rounds),
        partial_rounds(partial_rounds),
        alpha(alpha),
        ark(std::move(ark)) {}

  virtual ~PoseidonConfigBase() = default;

  virtual bool IsValid() const {
    return static_cast<size_t>(ark.rows()) == full_rounds + partial_rounds &&
           static_cast<size_t>(ark.cols()) == rate + capacity;
  }

  bool operator==(const PoseidonConfigBase& other) const {
    return SpongeConfig::operator==(other) &&
           full_rounds == other.full_rounds &&
           partial_rounds == other.partial_rounds && alpha == other.alpha &&
           ark == other.ark;
  }
  bool operator!=(const PoseidonConfigBase& other) const {
    return !operator==(other);
  }
};

}  // namespace crypto

namespace base {

template <typename F>
class Copyable<crypto::PoseidonConfigBase<F>> {
 public:
  static bool WriteTo(const crypto::PoseidonConfigBase<F>& config,
                      Buffer* buffer) {
    if (!Copyable<crypto::SpongeConfig>::WriteTo(config, buffer)) return false;
    return buffer->WriteMany(config.full_rounds, config.partial_rounds,
                             config.alpha, config.ark);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::PoseidonConfigBase<F>* config) {
    if (!Copyable<crypto::SpongeConfig>::ReadFrom(buffer, config)) return false;

    size_t full_rounds;
    size_t partial_rounds;
    uint64_t alpha;
    math::Matrix<F> ark;
    if (!buffer.ReadMany(&full_rounds, &partial_rounds, &alpha, &ark)) {
      return false;
    }

    config->full_rounds = full_rounds;
    config->partial_rounds = partial_rounds;
    config->alpha = alpha;
    config->ark = std::move(ark);
    return true;
  }

  static size_t EstimateSize(const crypto::PoseidonConfigBase<F>& config) {
    return Copyable<crypto::SpongeConfig>::EstimateSize(config) +
           base::EstimateSize(config.full_rounds, config.partial_rounds,
                              config.alpha, config.ark);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_BASE_H_
