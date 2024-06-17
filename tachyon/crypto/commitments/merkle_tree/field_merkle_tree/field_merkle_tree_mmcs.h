// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_MMCS_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_MMCS_H_

#include <utility>
#include <vector>

#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree.h"
#include "tachyon/crypto/commitments/mixed_matrix_commitment_scheme.h"

namespace tachyon::crypto {

template <typename PackedPrimeField, typename Hasher, typename PackedHasher,
          typename Compressor, typename PackedCompressor, size_t N>
class FieldMerkleTreeMMCS final
    : public MixedMatrixCommitmentScheme<
          FieldMerkleTreeMMCS<PackedPrimeField, Hasher, PackedHasher,
                              Compressor, PackedCompressor, N>> {
 public:
  using PrimeField = typename PackedPrimeField::PrimeField;
  using Commitment = std::array<PrimeField, N>;

  FieldMerkleTreeMMCS(const Hasher& hasher, const PackedHasher& packed_hasher,
                      const Compressor& compressor,
                      const PackedCompressor& packed_compressor)
      : hasher_(hasher),
        packed_hasher_(packed_hasher),
        compressor_(compressor),
        packed_compressor_(packed_compressor) {}
  FieldMerkleTreeMMCS(Hasher&& hasher, PackedHasher&& packed_hasher,
                      Compressor&& compressor,
                      PackedCompressor&& packed_compressor)
      : hasher_(std::move(hasher)),
        packed_hasher_(std::move(packed_hasher)),
        compressor_(std::move(compressor)),
        packed_compressor_(std::move(packed_compressor)) {}

  const FieldMerkleTree<PackedPrimeField, N>& field_merkle_tree() const {
    return field_merkle_tree_;
  }
  const Hasher& hasher() const { return hasher_; }
  const PackedHasher& packed_hasher() const { return packed_hasher_; }
  const Compressor& compressor() const { return compressor_; }
  const PackedCompressor& packed_compressor() const {
    return packed_compressor_;
  }

 private:
  friend class MixedMatrixCommitmentScheme<FieldMerkleTreeMMCS<
      PackedPrimeField, Hasher, PackedHasher, Compressor, PackedCompressor, N>>;

  [[nodiscard]] bool DoCommit(
      std::vector<math::RowMajorMatrix<PrimeField>>&& matrices,
      Commitment* result) {
    field_merkle_tree_ = FieldMerkleTree<PackedPrimeField, N>::Build(
        hasher_, packed_hasher_, compressor_, packed_compressor_,
        std::move(matrices));
    *result = field_merkle_tree_.GetRoot();
    return true;
  }

  FieldMerkleTree<PackedPrimeField, N> field_merkle_tree_;
  Hasher hasher_;
  PackedHasher packed_hasher_;
  Compressor compressor_;
  PackedCompressor packed_compressor_;
};

template <typename PackedPrimeField, typename Hasher, typename PackedHasher,
          typename Compressor, typename PackedCompressor, size_t N>
struct MixedMatrixCommitmentSchemeTraits<FieldMerkleTreeMMCS<
    PackedPrimeField, Hasher, PackedHasher, Compressor, PackedCompressor, N>> {
 public:
  using Field = typename PackedPrimeField::PrimeField;
  using Commitment = std::array<Field, N>;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_MMCS_H_
