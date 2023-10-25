// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_HALO2_POSEIDON_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_HALO2_POSEIDON_H_

#include <vector>

#include "third_party/gmp/include/gmpxx.h"

#include "tachyon/crypto/hashes/sponge/poseidon/poseidon.h"

namespace tachyon::crypto {

template <typename PrimeFieldTy>
struct Halo2PoseidonSponge : public PoseidonSponge<PrimeFieldTy> {
  using F = PrimeFieldTy;

  Halo2PoseidonSponge() = default;
  explicit Halo2PoseidonSponge(const PoseidonConfig<F>& config)
      : PoseidonSponge<PrimeFieldTy>(config) {
    this->state.elements[0] = F::FromMpzClass(mpz_class(1) << 64);
  }

  // FieldBasedCryptographicSponge methods
  std::vector<F> SqueezeNativeFieldElements(size_t num_elements) {
    std::vector<F> ret = base::CreateVector(num_elements, F::Zero());
    size_t squeeze_index = this->state.mode.next_index;
    if (squeeze_index == this->config.rate) {
      squeeze_index = 0;
    }
    this->state[squeeze_index + 1] = F::One();
    switch (this->state.mode.type) {
      case DuplexSpongeMode::Type::kAbsorbing: {
        this->Permute();
        this->SqueezeInternal(0, &ret);
        return ret;
      }
      case DuplexSpongeMode::Type::kSqueezing: {
        size_t squeeze_index = this->state.mode.next_index;
        if (squeeze_index == this->config.rate) {
          this->Permute();
          squeeze_index = 0;
        }
        this->SqueezeInternal(squeeze_index, &ret);
        return ret;
      }
    }
    NOTREACHED();
    return {};
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_POSEIDON_HALO2_POSEIDON_H_
