// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.lambdaworks.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_

#include <vector>

#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/binary_merkle_proof.h"

namespace tachyon::crypto {

template <typename F>
struct FRIProof {
  std::vector<BinaryMerkleProof<F, F>> proof;
  std::vector<BinaryMerkleProof<F, F>> proof_sym;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_FRI_PROOF_H_
