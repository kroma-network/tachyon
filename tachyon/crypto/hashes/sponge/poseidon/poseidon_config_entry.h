// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_ENTRY_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_ENTRY_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_config_entry_base.h"

namespace tachyon::crypto {

template <typename Params>
struct PoseidonConfig;

// An entry in the Poseidon config
struct TACHYON_EXPORT PoseidonConfigEntry : public PoseidonConfigEntryBase {
  // Number of matrices to skip when generating config using the Grain LFSR.
  // The matrices being skipped are those that do not satisfy all the desired
  // properties. See:
  // https://extgit.iaik.tugraz.at/krypto/hadeshash/-/blob/master/code/generate_parameters_grain.sage
  size_t skip_matrices;

  constexpr PoseidonConfigEntry() : PoseidonConfigEntry(0, 0, 0, 0, 0) {}
  constexpr PoseidonConfigEntry(size_t rate, uint64_t alpha, size_t full_rounds,
                                size_t partial_rounds, size_t skip_matrices)
      : PoseidonConfigEntryBase(rate, alpha, full_rounds, partial_rounds),
        skip_matrices(skip_matrices) {}

  template <typename Params>
  PoseidonConfig<Params> ToPoseidonConfig() const;

  bool operator==(const PoseidonConfigEntry& other) const {
    return PoseidonConfigEntryBase::operator==(other) &&
           skip_matrices == other.skip_matrices;
  }
  bool operator!=(const PoseidonConfigEntry& other) const {
    return !operator==(other);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_ENTRY_H_
