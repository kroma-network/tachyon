// Use of this source code is governed by a Apache-2.0 style license that
// can be found in the LICENSE.lambdaworks.

#include "tachyon/crypto/commitments/merkle_tree/binary_merkle_tree/binary_merkle_tree.h"

#include "gtest/gtest.h"

#include "tachyon/base/containers/container_util.h"

namespace tachyon::crypto {

namespace {

class SimpleHasher : public BinaryMerkleHasher<int, int> {
 public:
  // BinaryMerkleHasher<int, int> methods
  int ComputeLeafHash(const int& leaf) const override { return leaf; }
  int ComputeParentHash(const int& left, const int& right) const override {
    return left + 2 * right;
  }
};

class SimpleMerkleTreeStorage : public BinaryMerkleTreeStorage<int> {
 public:
  const std::vector<int>& hashes() const { return hashes_; }

  // BinaryMerkleTreeStorage<int> methods
  void Allocate(size_t size) override { hashes_.resize(size); }
  size_t GetSize() const override { return hashes_.size(); }
  const int& GetHash(size_t i) const override { return hashes_[i]; }
  void SetHash(size_t i, const int& hash) override { hashes_[i] = hash; }

 private:
  std::vector<int> hashes_;
};

class BinaryMerkleTreeTest : public testing::Test {
 public:
  constexpr static size_t K = 3;
  constexpr static size_t N = size_t{1} << K;

  using VCS = BinaryMerkleTree<int, int, N>;

  void SetUp() override {
    vcs_ = VCS(&storage_, &hasher_);
    vcs_.set_leaves_size_for_parallelization(N >> 1);
  };

  void CreateLeaves() { leaves_ = base::CreateRangedVector<int>(0, N); }

 protected:
  SimpleMerkleTreeStorage storage_;
  SimpleHasher hasher_;
  VCS vcs_;
  std::vector<int> leaves_;
};

}  // namespace

TEST_F(BinaryMerkleTreeTest, FillLeaves) {
  std::vector<int> invalid_leaves = base::CreateRangedVector<int>(0, N + 1);
  EXPECT_FALSE(vcs_.FillLeaves(invalid_leaves));
  invalid_leaves = base::CreateRangedVector<int>(0, size_t{1} << (K + 1));
  EXPECT_FALSE(vcs_.FillLeaves(invalid_leaves));
}

TEST_F(BinaryMerkleTreeTest, BuildTreeFromLeaves) {
  CreateLeaves();
  ASSERT_TRUE(vcs_.FillLeaves(leaves_));

  vcs_.BuildTreeFromLeaves(base::Range<size_t>(7, 11));
  // clang-format off
  std::vector<int> expected_nodes = {
    0,
    18, 0,
    2, 8, 0, 0,
    0, 1, 2, 3, 4, 5, 6, 7,
  };
  // clang-format on
  EXPECT_EQ(storage_.hashes(), expected_nodes);

  vcs_.BuildTreeFromLeaves(base::Range<size_t>(11, 15));
  // clang-format off
  expected_nodes = {
    0,
    18, 54,
    2, 8, 14, 20,
    0, 1, 2, 3, 4, 5, 6, 7,
  };
  // clang-format on
  EXPECT_EQ(storage_.hashes(), expected_nodes);

  vcs_.BuildTreeFromLeaves(base::Range<size_t>(1, 3));
  // clang-format off
  expected_nodes = {
    126,
    18, 54,
    2, 8, 14, 20,
    0, 1, 2, 3, 4, 5, 6, 7,
  };
  // clang-format on
  EXPECT_EQ(storage_.hashes(), expected_nodes);
}

TEST_F(BinaryMerkleTreeTest, CommitAndVerify) {
  CreateLeaves();

  int commitment;
  ASSERT_TRUE(vcs_.Commit(leaves_, &commitment));
  EXPECT_EQ(commitment, 126);

  BinaryMerkleProof<int> proof;
  ASSERT_TRUE(vcs_.CreateOpeningProof(1, &proof));

  BinaryMerkleProof<int> expected_proof;
  expected_proof.paths = std::vector<BinaryMerklePath<int>>{
      {true, 0},
      {false, 8},
      {false, 54},
  };
  EXPECT_EQ(proof, expected_proof);

  ASSERT_TRUE(vcs_.VerifyOpeningProof(commitment, 1, proof));
}

}  // namespace tachyon::crypto
