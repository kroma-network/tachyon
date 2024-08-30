// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.lambdaworks.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_SIMPLE_FRI_PROOF_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_SIMPLE_FRI_PROOF_H_

#include <vector>

#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/binary_merkle_proof.h"

namespace tachyon::crypto {

template <typename F>
struct SimpleFRIProof {
  std::vector<BinaryMerkleProof<F>> paths;
  std::vector<BinaryMerkleProof<F>> paths_sym;
  std::vector<F> evaluations;
  std::vector<F> evaluations_sym;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_SIMPLE_FRI_PROOF_H_
