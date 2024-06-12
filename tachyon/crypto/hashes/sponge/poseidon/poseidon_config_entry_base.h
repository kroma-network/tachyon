// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_ENTRY_BASE_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_ENTRY_BASE_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_grain_lfsr.h"

namespace tachyon::crypto {

// An entry in the Poseidon config
struct TACHYON_EXPORT PoseidonConfigEntryBase {
  // The rate (in terms of number of field elements).
  size_t rate;

  // Exponent used in S-boxes.
  uint64_t alpha;

  // Number of rounds in a full-round operation.
  size_t full_rounds;

  // Number of rounds in a partial-round operation.
  size_t partial_rounds;

  constexpr PoseidonConfigEntryBase() : PoseidonConfigEntryBase(0, 0, 0, 0) {}
  constexpr PoseidonConfigEntryBase(size_t rate, uint64_t alpha,
                                    size_t full_rounds, size_t partial_rounds)
      : rate(rate),
        alpha(alpha),
        full_rounds(full_rounds),
        partial_rounds(partial_rounds) {}

  template <typename F>
  PoseidonGrainLFSRConfig ToPoseidonGrainLFSRConfig() const {
    PoseidonGrainLFSRConfig config;
    config.prime_num_bits = F::kModulusBits;
    config.state_len = rate + 1;
    config.num_full_rounds = full_rounds;
    config.num_partial_rounds = partial_rounds;
    return config;
  }

  bool operator==(const PoseidonConfigEntryBase& other) const {
    return rate == other.rate && alpha == other.alpha &&
           full_rounds == other.full_rounds &&
           partial_rounds == other.partial_rounds;
  }
  bool operator!=(const PoseidonConfigEntryBase& other) const {
    return !operator==(other);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_ENTRY_BASE_H_
