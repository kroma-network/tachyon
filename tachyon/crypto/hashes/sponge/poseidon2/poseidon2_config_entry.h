// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_CONFIG_ENTRY_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_CONFIG_ENTRY_H_

#include <stddef.h>
#include <stdint.h>

#include "tachyon/crypto/hashes/sponge/poseidon/poseidon_config_entry_base.h"

namespace tachyon::crypto {

template <typename Params>
struct Poseidon2Config;

// An entry in the Poseidon config
struct TACHYON_EXPORT Poseidon2ConfigEntry : public PoseidonConfigEntryBase {
  using PoseidonConfigEntryBase::PoseidonConfigEntryBase;

  template <typename Params>
  Poseidon2Config<Params> ToPoseidon2Config() const;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON2_POSEIDON2_CONFIG_ENTRY_H_
