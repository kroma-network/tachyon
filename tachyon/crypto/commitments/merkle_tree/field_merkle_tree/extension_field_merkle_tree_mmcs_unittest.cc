// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/extension_field_merkle_tree_mmcs.h"

#include <memory>

#include "gtest/gtest.h"

#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree_mmcs.h"
#include "tachyon/crypto/hashes/sponge/padding_free_sponge.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_plonky3_external_matrix.h"
#include "tachyon/crypto/hashes/sponge/truncated_permutation.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear4.h"
#include "tachyon/math/finite_fields/baby_bear/packed_baby_bear.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"
#include "tachyon/math/finite_fields/test/finite_field_test.h"

namespace tachyon::crypto {

constexpr size_t kRate = 8;
constexpr size_t kChunk = 8;
constexpr size_t kN = 2;

using F = math::BabyBear;
using ExtF = math::BabyBear4;
using PackedF = math::PackedBabyBear;
using Poseidon2 =
    Poseidon2Sponge<Poseidon2ExternalMatrix<Poseidon2Plonky3ExternalMatrix<F>>>;
using PackedPoseidon2 = Poseidon2Sponge<
    Poseidon2ExternalMatrix<Poseidon2Plonky3ExternalMatrix<PackedF>>>;
using MyHasher = PaddingFreeSponge<Poseidon2, kRate, kChunk>;
using MyPackedHasher = PaddingFreeSponge<PackedPoseidon2, kRate, kChunk>;
using MyCompressor = TruncatedPermutation<Poseidon2, kChunk, kN>;
using MyPackedCompressor = TruncatedPermutation<PackedPoseidon2, kChunk, kN>;
using MMCS = FieldMerkleTreeMMCS<F, MyHasher, MyPackedHasher, MyCompressor,
                                 MyPackedCompressor, kChunk>;
using InnerMMCS = FieldMerkleTreeMMCS<ExtF, MyHasher, MyPackedHasher,
                                      MyCompressor, MyPackedCompressor, kChunk>;
using ExtMMCS = ExtensionFieldMerkleTreeMMCS<ExtF, InnerMMCS>;
using Tree = FieldMerkleTree<F, kChunk>;
using ExtTree = FieldMerkleTree<ExtF, kChunk>;

namespace {

class ExtensionFieldMerkleTreeMMCSTest : public math::FiniteFieldTest<PackedF> {
 public:
  static void SetUpTestSuite() {
    math::FiniteFieldTest<PackedF>::SetUpTestSuite();
    ExtF::Init();
  }

  void SetUp() override {
    Poseidon2Config<F> config = Poseidon2Config<F>::CreateCustom(
        15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
    Poseidon2 sponge(std::move(config));
    MyHasher hasher(sponge);
    MyCompressor compressor(std::move(sponge));

    Poseidon2Config<PackedF> packed_config =
        Poseidon2Config<PackedF>::CreateCustom(
            15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
    PackedPoseidon2 packed_sponge(std::move(packed_config));
    MyPackedHasher packed_hasher(packed_sponge);
    MyPackedCompressor packed_compressor(std::move(packed_sponge));
    ext_mmcs_.reset(new ExtMMCS(
        InnerMMCS(std::move(hasher), std::move(packed_hasher),
                  std::move(compressor), std::move(packed_compressor))));
  }

 protected:
  std::unique_ptr<ExtMMCS> ext_mmcs_;
};

}  // namespace

TEST_F(ExtensionFieldMerkleTreeMMCSTest, CommitAndVerify) {
  constexpr size_t kRows = 16;
  constexpr size_t kCols = 16;
  math::RowMajorMatrix<ExtF> ext_matrix =
      math::RowMajorMatrix<ExtF>::Random(kRows, kCols);

  math::RowMajorMatrix<F> matrix(kRows,
                                 kCols * ExtF::kDegreeOverBasePrimeField);
  for (size_t i = 0; i < kRows; ++i) {
    for (size_t j = 0; j < kCols * ExtF::kDegreeOverBasePrimeField; ++j) {
      size_t col = j / ExtF::kDegreeOverBasePrimeField;
      size_t idx = j % ExtF::kDegreeOverBasePrimeField;
      matrix(i, j) = ext_matrix(i, col)[idx];
    }
  }

  std::vector<math::RowMajorMatrix<ExtF>> ext_matrices = {
      std::move(ext_matrix)};
  std::array<F, kChunk> ext_commitment;
  ExtTree ext_prover_data;
  ASSERT_TRUE(ext_mmcs_->Commit(std::move(ext_matrices), &ext_commitment,
                                &ext_prover_data));

  const InnerMMCS& inner_mmcs = ext_mmcs_->inner();
  MMCS mmcs(inner_mmcs.hasher(), inner_mmcs.packed_hasher(),
            inner_mmcs.compressor(), inner_mmcs.packed_compressor());
  std::vector<math::RowMajorMatrix<F>> matrices = {std::move(matrix)};
  std::array<F, kChunk> commitment;
  Tree prover_data;
  ASSERT_TRUE(mmcs.Commit(std::move(matrices), &commitment, &prover_data));

  EXPECT_EQ(ext_commitment, commitment);

  size_t index = 5;
  std::vector<std::vector<ExtF>> openings;
  std::vector<std::array<F, kChunk>> proof;
  ASSERT_TRUE(
      ext_mmcs_->CreateOpeningProof(index, ext_prover_data, &openings, &proof));
  std::vector<math::Dimensions> dimensions_vec;
  dimensions_vec.push_back({kCols, kRows});
  ASSERT_TRUE(ext_mmcs_->VerifyOpeningProof(ext_commitment, dimensions_vec,
                                            index, openings, proof));
}

}  // namespace tachyon::crypto
