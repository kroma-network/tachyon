// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.lambdaworks.

#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_PROOF_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_PROOF_H_

#include <vector>

namespace tachyon::crypto {

template <typename HashTy>
struct BinaryMerklePath {
  bool left;
  HashTy hash;

  bool operator==(const BinaryMerklePath& other) const {
    return left == other.left && hash == other.hash;
  }
  bool operator!=(const BinaryMerklePath& other) const {
    return !operator==(other);
  }
};

template <typename HashTy>
struct BinaryMerkleProof {
  std::vector<BinaryMerklePath<HashTy>> paths;

  bool operator==(const BinaryMerkleProof& other) const {
    return paths == other.paths;
  }
  bool operator!=(const BinaryMerkleProof& other) const {
    return paths != other.paths;
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_PROOF_H_
