#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs_bn254.h"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "third_party/icicle/src/merkle-tree/merkle.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/merkle-tree/mmcs.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/poseidon/constants.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/poseidon2/constants.cu.cc"  // NOLINT(build/include)

#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs.h"

cudaError_t tachyon_bn254_mmcs_commit_cuda(
    const ::matrix::Matrix<::bn254::scalar_t>* leaves,
    unsigned int number_of_inputs, ::bn254::scalar_t* digests,
    const ::hash::Hasher<::bn254::scalar_t, ::bn254::scalar_t>* hasher,
    const ::hash::Hasher<::bn254::scalar_t, ::bn254::scalar_t>* compression,
    const ::merkle_tree::TreeBuilderConfig& tree_config) {
  return ::merkle_tree::mmcs_commit<::bn254::scalar_t, ::bn254::scalar_t>(
      leaves, number_of_inputs, digests, *hasher, *compression, tree_config);
}

namespace tachyon::crypto {

template <>
bool IcicleMMCS<math::bn254::Fr>::DoCommit(
    std::vector<math::RowMajorMatrix<math::bn254::Fr>>&& matrices) {
#if FIELD_ID != BN254
#error Only BN254 is supported
#endif
  /// Tree of height N and arity A contains \sum{A^i} for i in 0..N elements
  uint32_t tree_arity = 2;
  uint32_t input_block_len = 5;
  uint32_t digest_elements = 8;
  uint64_t tree_height = 2;
  uint64_t number_of_leaves = pow(tree_arity, tree_height);

  ::device_context::DeviceContext ctx =
      ::device_context::get_default_device_context();

  uint32_t width = 16;
  uint32_t rate = 8;
  ::poseidon2::Poseidon2<::bn254::scalar_t> poseidon(
      width, rate, ::poseidon2::MdsType::PLONKY,
      ::poseidon2::DiffusionStrategy::MONTGOMERY, ctx);

  /// Use keep_rows to specify how many rows do you want to store
  int keep_rows = 3;
  size_t digests_len = ::merkle_tree::get_digests_len(keep_rows - 1, tree_arity,
                                                      digest_elements);

  unsigned int number_of_inputs = matrices.size();
  ::matrix::Matrix<::bn254::scalar_t>* leaves =
      static_cast<::matrix::Matrix<::bn254::scalar_t>*>(malloc(
          number_of_inputs * sizeof(::matrix::Matrix<::bn254::scalar_t>)));

  size_t idx = 0;
  for (const auto& matrix : matrices) {
    uint64_t current_matrix_rows = matrix.rows();
    uint64_t current_matrix_size = matrix.size();

    absl::Span<const math::bn254::Fr> matrix_span =
        absl::Span<const math::bn254::Fr>(matrix.data(), matrix.size());
    auto reinterpret_cast_test =
        reinterpret_cast<const ::bn254::scalar_t*>(std::data(matrix_span));

    // Destination data (must have the same size)
    std::vector<::bn254::scalar_t> dest_data(matrix_span.size());
    absl::Span<::bn254::scalar_t> dest_span(dest_data);
    // Apply from_montgomery and move data
    for (size_t i = 0; i < matrix_span.size(); ++i) {
      dest_span[i] =
          ::bn254::scalar_t::from_montgomery(reinterpret_cast_test[i]);
    }

    ::bn254::scalar_t* d_matrix;
    cudaMalloc(&d_matrix, current_matrix_size * sizeof(::bn254::scalar_t));
    cudaMemcpy(d_matrix, dest_span.data(),
               current_matrix_size * sizeof(::bn254::scalar_t),
               cudaMemcpyHostToDevice);

    leaves[idx] = {
        d_matrix,
        input_block_len,
        current_matrix_rows,
    };
    ++idx;
  }

  /// Allocate memory for digests of {keep_rows} rows of a tree
  size_t digests_mem = digests_len * sizeof(::bn254::scalar_t);
  ::bn254::scalar_t* icicle_digest =
      static_cast<::bn254::scalar_t*>(malloc(digests_mem));

  ::merkle_tree::TreeBuilderConfig tree_config =
      ::merkle_tree::default_merkle_config();
  tree_config.are_inputs_on_device = true;
  tree_config.arity = tree_arity;
  tree_config.keep_rows = keep_rows;
  tree_config.digest_elements = digest_elements;

  cudaError_t error =
      tachyon_bn254_mmcs_commit_cuda(leaves, number_of_inputs, icicle_digest,
                                     &poseidon, &poseidon, tree_config);

  for (int i = 0; i < 10; i++) {
    std::cout << icicle_digest[digests_len - i - 1] << std::endl;
  }

  free(icicle_digest);
  free(leaves);
  bool result = false;
  if (error == cudaSuccess) {
    result = true;
  }
  return result;
}

bool DoCommitTest(
    std::vector<math::RowMajorMatrix<math::bn254::Fr>>&& matrices) {
#if FIELD_ID != BN254
#error Only BN254 is supported
#endif
  /// Tree of height N and arity A contains \sum{A^i} for i in 0..N elements
  uint32_t tree_arity = 2;
  uint32_t input_block_len = 5;
  uint32_t digest_elements = 8;
  uint64_t tree_height = 2;
  uint64_t number_of_leaves = pow(tree_arity, tree_height);

  ::device_context::DeviceContext ctx =
      ::device_context::get_default_device_context();

  uint32_t width = 16;
  uint32_t rate = 8;
  ::poseidon2::Poseidon2<::bn254::scalar_t> poseidon(
      width, rate, ::poseidon2::MdsType::PLONKY,
      ::poseidon2::DiffusionStrategy::MONTGOMERY, ctx);

  /// Use keep_rows to specify how many rows do you want to store
  int keep_rows = 3;
  size_t digests_len = ::merkle_tree::get_digests_len(keep_rows - 1, tree_arity,
                                                      digest_elements);

  // unsigned int number_of_inputs = tree_height * copied_matrices;
  unsigned int number_of_inputs = matrices.size();
  ::matrix::Matrix<::bn254::scalar_t>* leaves =
      static_cast<::matrix::Matrix<::bn254::scalar_t>*>(malloc(
          number_of_inputs * sizeof(::matrix::Matrix<::bn254::scalar_t>)));

  size_t idx = 0;
  for (const auto& matrix : matrices) {
    uint64_t current_matrix_rows = matrix.rows();
    uint64_t current_matrix_size = matrix.size();

    absl::Span<const math::bn254::Fr> matrix_span =
        absl::Span<const math::bn254::Fr>(matrix.data(), matrix.size());
    auto reinterpret_cast_test =
        reinterpret_cast<const ::bn254::scalar_t*>(std::data(matrix_span));

    // Destination data (must have the same size)
    std::vector<::bn254::scalar_t> dest_data(matrix_span.size());
    absl::Span<::bn254::scalar_t> dest_span(dest_data);
    // Apply from_montgomery and move data
    for (size_t i = 0; i < matrix_span.size(); ++i) {
      dest_span[i] =
          ::bn254::scalar_t::from_montgomery(reinterpret_cast_test[i]);
    }

    ::bn254::scalar_t* d_matrix;
    cudaMalloc(&d_matrix, current_matrix_size * sizeof(::bn254::scalar_t));
    cudaMemcpy(d_matrix, dest_span.data(),
               current_matrix_size * sizeof(::bn254::scalar_t),
               cudaMemcpyHostToDevice);

    leaves[idx] = {
        d_matrix,
        input_block_len,
        current_matrix_rows,
    };
    ++idx;
  }

  /// Allocate memory for digests of {keep_rows} rows of a tree
  size_t digests_mem = digests_len * sizeof(::bn254::scalar_t);
  ::bn254::scalar_t* icicle_digest =
      static_cast<::bn254::scalar_t*>(malloc(digests_mem));

  ::merkle_tree::TreeBuilderConfig tree_config =
      ::merkle_tree::default_merkle_config();
  tree_config.are_inputs_on_device = true;
  tree_config.arity = tree_arity;
  tree_config.keep_rows = keep_rows;
  tree_config.digest_elements = digest_elements;

  cudaError_t error =
      tachyon_bn254_mmcs_commit_cuda(leaves, number_of_inputs, icicle_digest,
                                     &poseidon, &poseidon, tree_config);

  for (int i = 0; i < 10; i++) {
    std::cout << icicle_digest[digests_len - i - 1] << std::endl;
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
