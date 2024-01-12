#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_SIMPLE_BINARY_MERKLE_TREE_STORAGE_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_SIMPLE_BINARY_MERKLE_TREE_STORAGE_H_

#include <vector>

#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/binary_merkle_tree_storage.h"

namespace tachyon::crypto {

template <typename Leaf, typename Hash>
class SimpleBinaryMerkleTreeStorage
    : public BinaryMerkleTreeStorage<Leaf, Hash> {
 public:
  const std::vector<Leaf>& leaves() const { return leaves_; }
  const std::vector<Hash>& hashes() const { return hashes_; }

  // BinaryMerkleTreeStorage<Hash> methods
  void AllocateLeaves(size_t size) override { leaves_.resize(size); }
  size_t GetLeavesSize() const override { return leaves_.size(); }
  const Hash& GetLeaf(size_t i) const override { return leaves_[i]; }
  void SetLeaf(size_t i, const Leaf& leaf) override { leaves_[i] = leaf; }

  void AllocateHashes(size_t size) override { hashes_.resize(size); }
  size_t GetHashesSize() const override { return hashes_.size(); }
  const Hash& GetHash(size_t i) const override { return hashes_[i]; }
  void SetHash(size_t i, const Hash& hash) override { hashes_[i] = hash; }

 private:
  std::vector<Leaf> leaves_;
  std::vector<Hash> hashes_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_SIMPLE_BINARY_MERKLE_TREE_STORAGE_H_
