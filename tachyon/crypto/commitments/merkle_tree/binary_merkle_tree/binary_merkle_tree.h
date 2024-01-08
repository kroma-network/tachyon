// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.lambdaworks.

#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_TREE_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_TREE_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "gtest/gtest_prod.h"

#include "tachyon/base/bits.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/numerics/checked_math.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/range.h"
#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/binary_merkle_hasher.h"
#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/binary_merkle_proof.h"
#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/binary_merkle_tree_storage.h"
#include "tachyon/crypto/commitments/vector_commitment_scheme.h"

namespace tachyon::crypto {

template <typename LeafTy, typename HashTy, size_t MaxSize>
class BinaryMerkleTree final
    : public VectorCommitmentScheme<BinaryMerkleTree<LeafTy, HashTy, MaxSize>> {
 public:
  constexpr static size_t kDefaultLeavesSizeForParallelization = 1024;

  BinaryMerkleTree() = default;
  BinaryMerkleTree(BinaryMerkleTreeStorage<HashTy>* storage,
                   BinaryMerkleHasher<LeafTy, HashTy>* hasher)
      : storage_(storage), hasher_(hasher) {}

  size_t leaves_size_for_parallelization() const {
    return leaves_size_for_parallelization_;
  }
  void set_leaves_size_for_parallelization(
      size_t leaves_size_for_parallelization) {
    CHECK(base::bits::IsPowerOfTwo(leaves_size_for_parallelization));
    leaves_size_for_parallelization_ = leaves_size_for_parallelization;
  }

 private:
  FRIEND_TEST(BinaryMerkleTreeTest, FillLeaves);
  FRIEND_TEST(BinaryMerkleTreeTest, BuildTreeFromLeaves);

  friend class VectorCommitmentScheme<
      BinaryMerkleTree<LeafTy, HashTy, MaxSize>>;

  // VectorCommitmentScheme methods
  size_t N() const { return MaxSize; }

  template <typename ContainerTy>
  [[nodiscard]] bool DoCommit(const ContainerTy& leaves, HashTy* out) const {
    if (!FillLeaves(leaves)) return false;

    // For instance, if |leaves_size_for_parallelization_| equals 4, the
    // subtrees with root indices 1 and 2 will be constructed.
    //
    //         0
    //    1          2
    //  3   4     5    6
    // 7 8 9 10 11 12 13 14
    //
    // Finally, the remaining tree should be constructed from leaves 1 and 2.
    size_t leaves_size = std::size(leaves);
    if (leaves_size > leaves_size_for_parallelization_) {
      OPENMP_PARALLEL_FOR(size_t i = 0; i < leaves_size;
                          i += leaves_size_for_parallelization_) {
        size_t from = leaves_size - 1 + i;
        size_t to = from + leaves_size_for_parallelization_;
        BuildTreeFromLeaves(base::Range<size_t>(from, to));
      }
      size_t i = base::bits::Log2Floor(leaves_size) -
                 base::bits::Log2Floor(leaves_size_for_parallelization_);
      BuildTreeFromLeaves(
          base::Range<size_t>((1 << i) - 1, (1 << (i + 1)) - 1));
    } else {
      BuildTreeFromLeaves(
          base::Range<size_t>(leaves_size - 1, (leaves_size << 1) - 1));
    }
    *out = storage_->GetHash(0);
    return true;
  }

  [[nodiscard]] bool DoCreateOpeningProof(
      size_t index, BinaryMerkleProof<HashTy>* proof) const {
    size_t size = storage_->GetSize();
    index = (size >> 1) + index;
    proof->paths.resize(base::bits::Log2Floor(size));
    size_t i = 0;
    while (index > 0) {
      BinaryMerklePath<HashTy> path;
      if (index % 2 == 0) {
        path.left = true;
        path.hash = storage_->GetHash(index - 1);
      } else {
        path.left = false;
        path.hash = storage_->GetHash(index + 1);
      }
      proof->paths[i++] = std::move(path);

      index = (index - 1) >> 1;
    }
    return true;
  }

  [[nodiscard]] bool DoVerifyOpeningProof(
      const HashTy& root, const HashTy& leaf_hash,
      const BinaryMerkleProof<HashTy>& proof) const {
    HashTy hash = leaf_hash;
    for (const BinaryMerklePath<HashTy>& path : proof.paths) {
      if (path.left) {
        hash = hasher_->ComputeParentHash(path.hash, hash);
      } else {
        hash = hasher_->ComputeParentHash(hash, path.hash);
      }
    }
    return hash == root;
  }

  template <typename ContainerTy>
  bool FillLeaves(const ContainerTy& leaves) const {
    size_t leaves_size = std::size(leaves);
    if (!base::bits::IsPowerOfTwo(leaves_size)) {
      LOG(ERROR) << leaves_size << " is not a power of two";
      return false;
    }
    if (leaves_size > MaxSize) {
      LOG(ERROR) << "Too many leaves";
      return false;
    }
    base::CheckedNumeric<size_t> n = leaves_size;
    storage_->Allocate(((n << 1) - 1).ValueOrDie());
    OPENMP_PARALLEL_FOR(size_t i = 0; i < leaves_size; ++i) {
      storage_->SetHash(leaves_size + i - 1,
                        hasher_->ComputeLeafHash(leaves[i]));
    }
    return true;
  }

  void BuildTreeFromLeaves(base::Range<size_t> range) const {
    while (range.GetSize() > 0) {
      for (size_t i = range.from; i < range.to; i += 2) {
        storage_->SetHash(i >> 1,
                          hasher_->ComputeParentHash(storage_->GetHash(i),
                                                     storage_->GetHash(i + 1)));
      }
      range = base::Range<size_t>(range.from >> 1, (range.to >> 1) - 1);
    }
  }

  // not owned
  mutable BinaryMerkleTreeStorage<HashTy>* storage_ = nullptr;
  // not owned
  BinaryMerkleHasher<LeafTy, HashTy>* hasher_ = nullptr;
  size_t leaves_size_for_parallelization_ =
      kDefaultLeavesSizeForParallelization;
};

template <typename LeafTy, typename HashTy, size_t MaxSize>
struct VectorCommitmentSchemeTraits<BinaryMerkleTree<LeafTy, HashTy, MaxSize>> {
 public:
  constexpr static size_t kMaxSize = MaxSize;
  constexpr static bool kIsTransparent = true;

  // TODO(chokobole): The result of Keccak256 is not a field. How can we handle
  // this?
  using Field = HashTy;
  using Commitment = HashTy;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_BINARY_MERKLE_TREE_BINARY_MERKLE_TREE_H_
