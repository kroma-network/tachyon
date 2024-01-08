// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_KZG_SHPLONK_PROOF_H_
#define TACHYON_CRYPTO_COMMITMENTS_KZG_SHPLONK_PROOF_H_

namespace tachyon::crypto {

template <typename C>
struct SHPlonkProof {
  C h;
  C q;

  bool operator==(const SHPlonkProof& other) const {
    return h == other.h && q == other.q;
  }
  bool operator!=(const SHPlonkProof& other) const {
    return !operator==(other);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_KZG_SHPLONK_PROOF_H_
