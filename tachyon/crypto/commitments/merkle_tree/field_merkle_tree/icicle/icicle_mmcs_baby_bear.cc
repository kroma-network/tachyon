#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs_baby_bear.h"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"
#include "third_party/icicle/src/merkle-tree/merkle.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/merkle-tree/mmcs.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/poseidon2/constants.cu.cc"  // NOLINT(build/include)

#include "tachyon/base/bits.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"

cudaError_t tachyon_babybear_mmcs_commit_cuda(
    const ::matrix::Matrix<::babybear::scalar_t>* leaves,
    unsigned int number_of_inputs, ::babybear::scalar_t* digests,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>* hasher,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>*
        compression,
    const ::merkle_tree::TreeBuilderConfig& tree_config) {
  return ::merkle_tree::mmcs_commit<::babybear::scalar_t, ::babybear::scalar_t>(
      leaves, number_of_inputs, digests, *hasher, *compression, tree_config);
}

namespace tachyon::crypto {

template <>
bool IcicleMMCS<math::BabyBear>::DoCommit(
    std::vector<math::RowMajorMatrix<math::BabyBear>>&& matrices,
    std::vector<std::vector<std::vector<math::BabyBear>>>&& outputs,
    absl::Span<const math::BabyBear> round_constants,
    absl::Span<const math::BabyBear> internal_matrix_diag) {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  // setting poseidon2
  ::babybear::scalar_t* round_constants_ptr =
      static_cast<::babybear::scalar_t*>(
          malloc(round_constants.size() * sizeof(::babybear::scalar_t)));

  for (uint32_t i = 0; i < round_constants.size(); i++) {
    round_constants_ptr[i] = ::babybear::scalar_t::from_montgomery(
        reinterpret_cast<const ::babybear::scalar_t*>(
            std::data(round_constants))[i]);
  }

  ::babybear::scalar_t* internal_matrix_diag_ptr =
      static_cast<::babybear::scalar_t*>(
          malloc(internal_matrix_diag.size() * sizeof(::babybear::scalar_t)));

  for (uint32_t i = 0; i < internal_matrix_diag.size(); i++) {
    internal_matrix_diag_ptr[i] = ::babybear::scalar_t::from_montgomery(
        reinterpret_cast<const ::babybear::scalar_t*>(
            std::data(internal_matrix_diag))[i]);
  }

  ::poseidon2::Poseidon2<::babybear::scalar_t> icicle_poseidon(
      16, 8, 7, 13, 8, round_constants_ptr, internal_matrix_diag_ptr,
      ::poseidon2::MdsType::PLONKY, ::poseidon2::DiffusionStrategy::MONTGOMERY,
      config_->ctx);

  // setting leaves
  size_t max_tree_height = 0;
  size_t number_of_leaves = 0;
  for (const auto& matrix : matrices) {
    size_t tree_height = base::bits::Log2Ceiling(
        absl::bit_ceil(static_cast<size_t>(matrix.rows())));
    max_tree_height =
        max_tree_height > tree_height ? max_tree_height : tree_height;
    number_of_leaves += matrix.size();
  }

  ::matrix::Matrix<::babybear::scalar_t>* leaves =
      static_cast<::matrix::Matrix<::babybear::scalar_t>*>(malloc(
          number_of_leaves * sizeof(::matrix::Matrix<::babybear::scalar_t>)));

  size_t idx = 0;
  for (const auto& matrix : matrices) {
    uint64_t current_matrix_size = matrix.size();

    absl::Span<const math::BabyBear> matrix_span =
        absl::Span<const math::BabyBear>(matrix.data(), matrix.size());
    auto reinterpret_cast_test =
        reinterpret_cast<const ::babybear::scalar_t*>(std::data(matrix_span));

    // Destination data (must have the same size)
    std::vector<::babybear::scalar_t> dest_data(matrix_span.size());
    absl::Span<::babybear::scalar_t> dest_span(dest_data);
    // Apply from_montgomery and move data
    for (size_t i = 0; i < matrix_span.size(); ++i) {
      dest_span[i] =
          ::babybear::scalar_t::from_montgomery(reinterpret_cast_test[i]);
    }

    ::babybear::scalar_t* d_matrix;
    cudaMalloc(&d_matrix, current_matrix_size * sizeof(::babybear::scalar_t));
    cudaMemcpy(d_matrix, dest_span.data(),
               current_matrix_size * sizeof(::babybear::scalar_t),
               cudaMemcpyHostToDevice);

    leaves[idx] = {
        d_matrix,
        static_cast<size_t>(matrix.cols()),
        static_cast<size_t>(matrix.rows()),
    };
    ++idx;
  }

  config_->keep_rows = max_tree_height + 1;
  config_->digest_elements = 8;
  size_t digests_len = ::merkle_tree::get_digests_len(
      config_->keep_rows - 1, config_->arity, config_->digest_elements);

  ::babybear::scalar_t* icicle_digest = static_cast<::babybear::scalar_t*>(
      malloc(digests_len * sizeof(::babybear::scalar_t)));

  cudaError_t error = tachyon_babybear_mmcs_commit_cuda(
      leaves, matrices.size(), icicle_digest, &icicle_poseidon,
      &icicle_poseidon, *config_);

  outputs.reserve(config_->keep_rows);
  size_t previous_number_of_element = 0;
  for (size_t layer_idx = 0; layer_idx <= max_tree_height; ++layer_idx) {
    std::vector<std::vector<math::BabyBear>> digest_layer;
    size_t number_of_node = 1 << (max_tree_height - layer_idx);
    digest_layer.reserve(number_of_node);
    for (size_t node_idx = 0; node_idx < number_of_node; ++node_idx) {
      std::vector<math::BabyBear> digest;
      digest.reserve(config_->digest_elements);
      for (size_t element_idx = 0; element_idx < config_->digest_elements;
           ++element_idx) {
        size_t idx = previous_number_of_element +
                     config_->digest_elements * node_idx + element_idx;
        *(icicle_digest + idx) =
            ::babybear::scalar_t::to_montgomery(*(icicle_digest + idx));
        digest.emplace_back(
            *reinterpret_cast<math::BabyBear*>(icicle_digest + idx));
      }
      digest_layer.emplace_back(std::move(digest));
    }
    outputs.emplace_back(std::move(digest_layer));
    previous_number_of_element += number_of_node * config_->digest_elements;
  }

  free(icicle_digest);
  free(leaves);
  bool result = false;
  if (error == cudaSuccess) {
    result = true;
  }
  return result;
}

}  // namespace tachyon::crypto
