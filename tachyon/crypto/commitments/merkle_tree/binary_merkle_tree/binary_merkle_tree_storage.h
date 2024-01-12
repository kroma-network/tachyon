// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.lambdaworks.

#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_TREE_STORAGE_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_TREE_STORAGE_H_

#include <stddef.h>

namespace tachyon::crypto {

template <typename Leaf, typename Hash>
class BinaryMerkleTreeStorage {
 public:
  virtual ~BinaryMerkleTreeStorage() = default;

  virtual void AllocateLeaves(size_t size) = 0;
  virtual size_t GetLeavesSize() const = 0;
  virtual const Leaf& GetLeaf(size_t i) const = 0;
  virtual void SetLeaf(size_t i, const Leaf& leaf) = 0;

  virtual void AllocateHashes(size_t size) = 0;
  virtual size_t GetHashesSize() const = 0;
  virtual const Hash& GetHash(size_t i) const = 0;
  virtual void SetHash(size_t i, const Hash& hash) = 0;

  Hash GetRoot() const { return GetHash(0); }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_TREE_STORAGE_H_
