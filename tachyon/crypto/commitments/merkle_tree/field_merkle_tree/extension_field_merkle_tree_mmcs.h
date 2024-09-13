// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_EXTENSION_FIELD_MERKLE_TREE_MMCS_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_EXTENSION_FIELD_MERKLE_TREE_MMCS_H_

#include <utility>
#include <vector>

#include "tachyon/crypto/commitments/mixed_matrix_commitment_scheme.h"

namespace tachyon::crypto {

template <typename ExtF, typename InnerMMCS>
class ExtensionFieldMerkleTreeMMCS final
    : public MixedMatrixCommitmentScheme<
          ExtensionFieldMerkleTreeMMCS<ExtF, InnerMMCS>> {
 public:
  using Commitment =
      typename MixedMatrixCommitmentSchemeTraits<InnerMMCS>::Commitment;
  using ProverData =
      typename MixedMatrixCommitmentSchemeTraits<InnerMMCS>::ProverData;
  using Digest = Commitment;
  using Proof = std::vector<Digest>;

  ExtensionFieldMerkleTreeMMCS() = default;
  ExtensionFieldMerkleTreeMMCS(InnerMMCS&& inner) : inner_(std::move(inner)) {}

  const InnerMMCS& inner() const { return inner_; }

 private:
  friend class MixedMatrixCommitmentScheme<
      ExtensionFieldMerkleTreeMMCS<ExtF, InnerMMCS>>;

  [[nodiscard]] bool DoCommit(
      std::vector<Eigen::Map<const math::RowMajorMatrix<ExtF>>>&& matrices,
      Commitment* commitment, ProverData* prover_data) const {
    return inner_.Commit(std::move(matrices), commitment, prover_data);
  }

  [[nodiscard]] bool DoCommitOwned(
      std::vector<math::RowMajorMatrix<ExtF>>&& owned_matrices,
      Commitment* commitment, ProverData* prover_data) const {
    return inner_.CommitOwned(std::move(owned_matrices), commitment,
                              prover_data);
  }

  const std::vector<Eigen::Map<const math::RowMajorMatrix<ExtF>>>&
  DoGetMatrices(const ProverData& prover_data) const {
    return prover_data.leaves();
  }

  [[nodiscard]] bool DoCreateOpeningProof(
      size_t index, const ProverData& prover_data,
      std::vector<std::vector<ExtF>>* openings, Proof* proof) const {
    return inner_.CreateOpeningProof(index, prover_data, openings, proof);
  }

  [[nodiscard]] bool DoVerifyOpeningProof(
      const Commitment& commitment,
      absl::Span<const math::Dimensions> dimensions_list, size_t index,
      absl::Span<const std::vector<ExtF>> opened_values,
      const Proof& proof) const {
    return inner_.VerifyOpeningProof(commitment, dimensions_list, index,
                                     opened_values, proof);
  }

  InnerMMCS inner_;
};

template <typename ExtF, typename InnerMMCS>
struct MixedMatrixCommitmentSchemeTraits<
    ExtensionFieldMerkleTreeMMCS<ExtF, InnerMMCS>> {
 public:
  using Field = ExtF;
  using Commitment =
      typename MixedMatrixCommitmentSchemeTraits<InnerMMCS>::Commitment;
  using ProverData =
      typename MixedMatrixCommitmentSchemeTraits<InnerMMCS>::ProverData;
  using Proof = typename MixedMatrixCommitmentSchemeTraits<InnerMMCS>::Proof;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_EXTENSION_FIELD_MERKLE_TREE_MMCS_H_
