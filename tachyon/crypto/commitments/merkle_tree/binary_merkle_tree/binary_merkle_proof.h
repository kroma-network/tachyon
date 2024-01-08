// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.lambdaworks.

#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_PROOF_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_PROOF_H_

#include <vector>

namespace tachyon::crypto {

template <typename Hash>
struct BinaryMerklePath {
  bool left;
  Hash hash;

  bool operator==(const BinaryMerklePath& other) const {
    return left == other.left && hash == other.hash;
  }
  bool operator!=(const BinaryMerklePath& other) const {
    return !operator==(other);
  }
};

template <typename Leaf, typename Hash>
struct BinaryMerkleProof {
  Leaf value;
  std::vector<BinaryMerklePath<Hash>> paths;

  bool operator==(const BinaryMerkleProof& other) const {
    return value == other.value && paths == other.paths;
  }
  bool operator!=(const BinaryMerkleProof& other) const {
    return !operator==(other);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_PROOF_H_
