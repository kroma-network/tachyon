// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree_mmcs.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/crypto/hashes/sponge/padding_free_sponge.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/crypto/hashes/sponge/truncated_permutation.h"
#include "tachyon/math/finite_fields/baby_bear/packed_baby_bear.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

constexpr size_t kRate = 8;
constexpr size_t kChunk = 8;
constexpr size_t kN = 2;

using F = math::BabyBear;
using PackedF = math::PackedBabyBear;
using Poseidon2 =
    Poseidon2Sponge<Poseidon2ExternalMatrix<Poseidon2Plonky3ExternalMatrix<F>>>;
using PackedPoseidon2 = Poseidon2Sponge<
    Poseidon2ExternalMatrix<Poseidon2Plonky3ExternalMatrix<PackedF>>>;
using MyHasher = PaddingFreeSponge<Poseidon2, kRate, kChunk>;
using MyPackedHasher = PaddingFreeSponge<PackedPoseidon2, kRate, kChunk>;
using MyCompressor = TruncatedPermutation<Poseidon2, kChunk, kN>;
using MyPackedCompressor = TruncatedPermutation<PackedPoseidon2, kChunk, kN>;
using Tree = FieldMerkleTree<PackedF, kChunk>;
using MMCS = FieldMerkleTreeMMCS<PackedF, MyHasher, MyPackedHasher,
                                 MyCompressor, MyPackedCompressor, kChunk>;

namespace {

class FieldMerkleTreeMMCSTest : public math::FiniteFieldTest<PackedF> {
 public:
  void SetUp() override {
    Poseidon2Config<F> config = Poseidon2Config<F>::CreateCustom(
        15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
    Poseidon2 sponge(config);
    MyHasher hasher(sponge);
    MyCompressor compressor(std::move(sponge));

    Poseidon2Config<PackedF> packed_config =
        Poseidon2Config<PackedF>::CreateCustom(
            15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
    PackedPoseidon2 packed_sponge(packed_config);
    MyPackedHasher packed_hasher(packed_sponge);
    MyPackedCompressor packed_compressor(packed_sponge);
    mmcs_.reset(new MMCS(hasher, packed_hasher, compressor, packed_compressor));
  }

 protected:
  std::unique_ptr<MMCS> mmcs_;
};

}  // namespace

TEST_F(FieldMerkleTreeMMCSTest, Commit) {
  std::vector<F> vector{F(2), F(1), F(2), F(2), F(0), F(0), F(1), F(0)};
  math::RowMajorMatrix<F> matrix{
      {F(2)}, {F(1)}, {F(2)}, {F(2)}, {F(0)}, {F(0)}, {F(1)}, {F(0)},
  };
  std::vector<math::RowMajorMatrix<F>> matrices = {matrix};

  std::vector<math::RowMajorMatrix<F>> matrices_tmp = matrices;
  Tree tree =
      Tree::Build(mmcs_->hasher(), mmcs_->packed_hasher(), mmcs_->compressor(),
                  mmcs_->packed_compressor(), std::move(matrices_tmp));

  {
    std::array<F, kChunk> commitment;
    ASSERT_TRUE(mmcs_->Commit(vector, &commitment));
    EXPECT_EQ(commitment, tree.GetRoot());
  }
  {
    std::array<F, kChunk> commitment;
    ASSERT_TRUE(mmcs_->Commit(std::move(matrix), &commitment));
    EXPECT_EQ(commitment, tree.GetRoot());
  }
  {
    std::array<F, kChunk> commitment;
    ASSERT_TRUE(mmcs_->Commit(std::move(matrices), &commitment));
    EXPECT_EQ(commitment, tree.GetRoot());
  }
}

TEST_F(FieldMerkleTreeMMCSTest, CommitAndVerify) {
  struct Config {
    size_t num;
    math::Dimensions dimensions;
  };

  struct {
    std::vector<Config> configs;
    size_t index;
  } tests[] = {
      {{{4, {8, 1000}}, {5, {8, 70}}, {6, {8, 8}}}, 6},
      {{{1, {1, 32}},
        {1, {2, 32}},
        {1, {3, 32}},
        {1, {4, 32}},
        {1, {5, 32}},
        {1, {6, 32}},
        {1, {7, 32}},
        {1, {8, 32}},
        {1, {9, 32}},
        {1, {10, 32}}},
       17},
  };

  for (const auto& test : tests) {
    std::vector<math::RowMajorMatrix<F>> matrices;
    std::vector<math::Dimensions> dimensions_vec;
    for (size_t i = 0; i < test.configs.size(); ++i) {
      math::Dimensions dimensions = test.configs[i].dimensions;
      for (size_t j = 0; j < test.configs[i].num; ++j) {
        matrices.push_back(math::RowMajorMatrix<F>::Random(dimensions.height,
                                                           dimensions.width));
        dimensions_vec.push_back(dimensions);
      }
    }

    std::array<F, kChunk> commitment;
    ASSERT_TRUE(mmcs_->Commit(std::move(matrices), &commitment));

    std::vector<std::vector<F>> openings;
    std::vector<std::array<F, kChunk>> proof;
    ASSERT_TRUE(mmcs_->CreateOpeningProof(test.index, &openings, &proof));
    ASSERT_TRUE(mmcs_->VerifyOpeningProof(commitment, dimensions_vec,
                                          test.index, openings, proof));
  }
}

}  // namespace tachyon::crypto
