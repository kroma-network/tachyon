#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_SIMPLE_BINARY_MERKLE_TREE_STORAGE_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_SIMPLE_BINARY_MERKLE_TREE_STORAGE_H_

#include <vector>

#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/binary_merkle_tree_storage.h"

namespace tachyon::crypto {

template <typename T>
class SimpleBinaryMerkleTreeStorage : public BinaryMerkleTreeStorage<T> {
 public:
  const std::vector<T>& hashes() const { return hashes_; }

  // BinaryMerkleTreeStorage<T> methods
  void Allocate(size_t size) override { hashes_.resize(size); }
  size_t GetSize() const override { return hashes_.size(); }
  const T& GetHash(size_t i) const override { return hashes_[i]; }
  void SetHash(size_t i, const T& hash) override { hashes_[i] = hash; }

 private:
  std::vector<T> hashes_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_SIMPLE_BINARY_MERKLE_TREE_STORAGE_H_
