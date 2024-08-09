// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_MMCS_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_MMCS_H_

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/field_merkle_tree.h"
#include "tachyon/crypto/commitments/mixed_matrix_commitment_scheme.h"
#include "tachyon/math/finite_fields/extension_field_traits_forward.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"

namespace tachyon::crypto {

template <typename F, typename Hasher, typename PackedHasher,
          typename Compressor, typename PackedCompressor, size_t N>
class FieldMerkleTreeMMCS final
    : public MixedMatrixCommitmentScheme<FieldMerkleTreeMMCS<
          F, Hasher, PackedHasher, Compressor, PackedCompressor, N>> {
 public:
  using PrimeField =
      std::conditional_t<math::FiniteFieldTraits<F>::kIsExtensionField,
                         typename math::ExtensionFieldTraits<F>::BasePrimeField,
                         F>;
  using Commitment = std::array<PrimeField, N>;
  using Digest = Commitment;
  using ProverData = FieldMerkleTree<F, N>;
  using Proof = std::vector<Digest>;

  FieldMerkleTreeMMCS() = default;
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

  const Hasher& hasher() const { return hasher_; }
  const PackedHasher& packed_hasher() const { return packed_hasher_; }
  const Compressor& compressor() const { return compressor_; }
  const PackedCompressor& packed_compressor() const {
    return packed_compressor_;
  }

 private:
  friend class MixedMatrixCommitmentScheme<FieldMerkleTreeMMCS<
      F, Hasher, PackedHasher, Compressor, PackedCompressor, N>>;

  struct IndexedDimensions {
    size_t index;
    math::Dimensions dimensions;

    // TODO(chokobole): This comparison is intentionally reversed to sort in
    // descending order, as powersort doesn't accept custom callbacks.
    bool operator<(const IndexedDimensions& other) const {
      return dimensions.height > other.dimensions.height;
    }
    bool operator<=(const IndexedDimensions& other) const {
      return dimensions.height >= other.dimensions.height;
    }
    bool operator>(const IndexedDimensions& other) const {
      return dimensions.height < other.dimensions.height;
    }

    std::string ToString() const {
      return absl::Substitute("($0, $1)", index, dimensions.ToString());
    }
  };

  [[nodiscard]] bool DoCommit(std::vector<math::RowMajorMatrix<F>>&& matrices,
                              Commitment* commitment, ProverData* prover_data) {
    *prover_data =
        FieldMerkleTree<F, N>::Build(hasher_, packed_hasher_, compressor_,
                                     packed_compressor_, std::move(matrices));
    *commitment = prover_data->GetRoot();

    return true;
  }

  const std::vector<math::RowMajorMatrix<F>>& DoGetMatrices(
      const ProverData& prover_data) const {
    return prover_data.leaves();
  }

  [[nodiscard]] bool DoCreateOpeningProof(size_t index,
                                          const ProverData& prover_data,
                                          std::vector<std::vector<F>>* openings,
                                          Proof* proof) const {
    size_t max_row_size = this->GetMaxRowSize(prover_data);
    uint32_t log_max_row_size = base::bits::Log2Ceiling(max_row_size);

    // TODO(chokobole): Is it able to be parallelized?
    *openings = base::Map(
        prover_data.leaves(),
        [log_max_row_size, index](const math::RowMajorMatrix<F>& matrix) {
          uint32_t log_row_size =
              base::bits::Log2Ceiling(static_cast<size_t>(matrix.rows()));
          uint32_t bits_reduced = log_max_row_size - log_row_size;
          size_t reduced_index = index >> bits_reduced;
          return base::CreateVector(matrix.cols(),
                                    [reduced_index, &matrix](size_t col) {
                                      return matrix(reduced_index, col);
                                    });
        });

    *proof =
        base::CreateVector(log_max_row_size, [prover_data, index](size_t i) {
          // NOTE(chokobole): Let v be |index >> i|. If v is even, v ^ 1 is v
          // + 1. Otherwise, v ^ 1 is v - 1.
          return prover_data.digest_layers()[i][(index >> i) ^ 1];
        });

    return true;
  }

  [[nodiscard]] bool DoVerifyOpeningProof(
      const Commitment& commitment,
      absl::Span<const math::Dimensions> dimensions_list, size_t index,
      absl::Span<const std::vector<F>> opened_values,
      const Proof& proof) const {
    CHECK_EQ(dimensions_list.size(), opened_values.size());

    std::vector<IndexedDimensions> sorted_dimensions_list = base::Map(
        dimensions_list, [](size_t index, math::Dimensions dimensions) {
          return IndexedDimensions{index, dimensions};
        });

    base::StableSort(sorted_dimensions_list.begin(),
                     sorted_dimensions_list.end());

    absl::Span<const IndexedDimensions> remaining_dimensions_list =
        absl::MakeConstSpan(sorted_dimensions_list);

    size_t next_layer =
        absl::bit_ceil(remaining_dimensions_list.front().dimensions.height);
    size_t next_layer_size = CountLayers(next_layer, remaining_dimensions_list);
    Digest root = hasher_.Hash(GetOpenedValuesAsPrimeFieldVectors(
        opened_values, remaining_dimensions_list.subspan(0, next_layer_size)));
    remaining_dimensions_list.remove_prefix(next_layer_size);

    for (const Digest& sibling : proof) {
      Digest inputs[2];
      inputs[0] = (index & 1) == 0 ? root : sibling;
      inputs[1] = (index & 1) == 0 ? sibling : root;
      root = compressor_.Compress(inputs);

      index >>= 1;
      next_layer >>= 1;
      next_layer_size = CountLayers(next_layer, remaining_dimensions_list);
      if (next_layer_size > 0) {
        inputs[0] = std::move(root);
        inputs[1] = hasher_.Hash(GetOpenedValuesAsPrimeFieldVectors(
            opened_values,
            remaining_dimensions_list.subspan(0, next_layer_size)));
        remaining_dimensions_list.remove_prefix(next_layer_size);

        root = compressor_.Compress(inputs);
      }
    }

    return root == commitment;
  }

  constexpr static size_t CountLayers(
      size_t target_height,
      absl::Span<const IndexedDimensions> dimensions_list) {
    size_t ret = 0;
    for (size_t i = 0; i < dimensions_list.size(); ++i) {
      if (target_height == absl::bit_ceil(static_cast<size_t>(
                               dimensions_list[i].dimensions.height))) {
        ++ret;
      } else {
        break;
      }
    }
    return ret;
  }

  static std::vector<PrimeField> GetOpenedValuesAsPrimeFieldVectors(
      absl::Span<const std::vector<F>> opened_values,
      absl::Span<const IndexedDimensions> dimensions_list) {
    if constexpr (math::FiniteFieldTraits<F>::kIsExtensionField) {
      static_assert(math::ExtensionFieldTraits<F>::kDegreeOverBasePrimeField ==
                    math::ExtensionFieldTraits<F>::kDegreeOverBaseField);
      size_t size = std::accumulate(
          dimensions_list.begin(), dimensions_list.end(), 0,
          [&opened_values](size_t acc, const IndexedDimensions& dimensions) {
            return acc + opened_values[dimensions.index].size();
          });
      std::vector<PrimeField> ret;
      ret.reserve(size *
                  math::ExtensionFieldTraits<F>::kDegreeOverBasePrimeField);
      for (size_t i = 0; i < dimensions_list.size(); ++i) {
        const std::vector<F>& elements =
            opened_values[dimensions_list[i].index];
        for (size_t j = 0; j < elements.size(); ++j) {
          const F& element = elements[j];
          for (size_t k = 0;
               k < math::ExtensionFieldTraits<F>::kDegreeOverBasePrimeField;
               ++k) {
            ret.push_back(element[k]);
          }
        }
      }
      return ret;
    } else {
      return base::FlatMap(
          dimensions_list,
          [&opened_values](const IndexedDimensions& dimensions) {
            return opened_values[dimensions.index];
          });
    }
  }

  Hasher hasher_;
  PackedHasher packed_hasher_;
  Compressor compressor_;
  PackedCompressor packed_compressor_;
};

template <typename F, typename Hasher, typename PackedHasher,
          typename Compressor, typename PackedCompressor, size_t N>
struct MixedMatrixCommitmentSchemeTraits<FieldMerkleTreeMMCS<
    F, Hasher, PackedHasher, Compressor, PackedCompressor, N>> {
 public:
  using Field = F;
  using PrimeField =
      std::conditional_t<math::FiniteFieldTraits<F>::kIsExtensionField,
                         typename math::ExtensionFieldTraits<F>::BasePrimeField,
                         F>;
  using Commitment = std::array<PrimeField, N>;
  using ProverData = FieldMerkleTree<F, N>;
  using Proof = std::vector<std::array<PrimeField, N>>;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_FIELD_MERKLE_TREE_MMCS_H_
