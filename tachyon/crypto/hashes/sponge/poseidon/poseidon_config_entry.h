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

template <typename PrimeField>
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

  template <typename PrimeField>
  PoseidonConfig<PrimeField> ToPoseidonConfig() const;

  bool operator==(const PoseidonConfigEntry& other) const {
    return PoseidonConfigEntryBase::operator==(other) &&
           skip_matrices == other.skip_matrices;
  }
  bool operator!=(const PoseidonConfigEntry& other) const {
    return !operator==(other);
  }
};

// An array of the default config optimized for constraints
// (rate, alpha, full_rounds, partial_rounds, skip_matrices)
// for rate = 2, 3, 4, 5, 6, 7, 8
// Here, |skip_matrices| denotes how many matrices to skip before finding one
// that satisfy all the requirements.
constexpr const PoseidonConfigEntry kOptimizedConstraintsDefaultParams[] = {
    PoseidonConfigEntry(2, 17, 8, 31, 0), PoseidonConfigEntry(3, 5, 8, 56, 0),
    PoseidonConfigEntry(4, 5, 8, 56, 0),  PoseidonConfigEntry(5, 5, 8, 57, 0),
    PoseidonConfigEntry(6, 5, 8, 57, 0),  PoseidonConfigEntry(7, 5, 8, 57, 0),
    PoseidonConfigEntry(8, 5, 8, 57, 0),
};

// An array of the default config optimized for weights
// (rate, alpha, full_rounds, partial_rounds, skip_matrices)
// for rate = 2, 3, 4, 5, 6, 7, 8
constexpr const PoseidonConfigEntry kOptimizedWeightsDefaultParams[] = {
    PoseidonConfigEntry(2, 257, 8, 13, 0),
    PoseidonConfigEntry(3, 257, 8, 13, 0),
    PoseidonConfigEntry(4, 257, 8, 13, 0),
    PoseidonConfigEntry(5, 257, 8, 13, 0),
    PoseidonConfigEntry(6, 257, 8, 13, 0),
    PoseidonConfigEntry(7, 257, 8, 13, 0),
    PoseidonConfigEntry(8, 257, 8, 13, 0),
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_POSEIDON_CONFIG_ENTRY_H_
