#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs_bls12_381.h"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "third_party/icicle/src/merkle-tree/merkle.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/merkle-tree/mmcs.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/poseidon/constants.cu.cc"  // NOLINT(build/include)

#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs.h"

cudaError_t tachyon_bls12_381_mmcs_commit_cuda(
    const ::matrix::Matrix<::bls12_381::scalar_t>* leaves,
    unsigned int number_of_inputs, ::bls12_381::scalar_t* digests,
    const ::hash::Hasher<::bls12_381::scalar_t, ::bls12_381::scalar_t>* hasher,
    const ::hash::Hasher<::bls12_381::scalar_t, ::bls12_381::scalar_t>*
        compression,
    const ::merkle_tree::TreeBuilderConfig& tree_config) {
  return ::merkle_tree::mmcs_commit<::bls12_381::scalar_t,
                                    ::bls12_381::scalar_t>(
      leaves, number_of_inputs, digests, *hasher, *compression, tree_config);
}

namespace tachyon::crypto {

template <>
bool IcicleMMCS<math::bls12_381::Fr>::DoCommit(
    std::vector<math::RowMajorMatrix<math::bls12_381::Fr>>&& matrices) {
#if FIELD_ID != BLS12_381
#error Only BLS12_381 is supported
#endif
  /// Tree of height N and arity A contains \sum{A^i} for i in 0..N elements
  uint32_t digest_elements = 8;
  uint64_t tree_height = 2;
  uint64_t number_of_leaves = pow(config_->arity, tree_height);
  int keep_rows = 3;

  config_->keep_rows = keep_rows;
  config_->digest_elements = digest_elements;
  ::device_context::DeviceContext ctx =
      ::device_context::get_default_device_context();

  ::poseidon::Poseidon<::bls12_381::scalar_t> poseidon(config_->arity, ctx);

  /// Use keep_rows to specify how many rows do you want to store
  size_t digests_len = ::merkle_tree::get_digests_len(
      config_->keep_rows - 1, config_->arity, config_->digest_elements);

  unsigned int number_of_inputs = matrices.size();
  ::matrix::Matrix<::bls12_381::scalar_t>* leaves =
      static_cast<::matrix::Matrix<::bls12_381::scalar_t>*>(malloc(
          number_of_inputs * sizeof(::matrix::Matrix<::bls12_381::scalar_t>)));

  size_t idx = 0;
  for (const auto& matrix : matrices) {
    uint64_t current_matrix_size = matrix.size();

    absl::Span<const math::bls12_381::Fr> matrix_span =
        absl::Span<const math::bls12_381::Fr>(matrix.data(), matrix.size());
    auto reinterpret_cast_test =
        reinterpret_cast<const ::bls12_381::scalar_t*>(std::data(matrix_span));

    // Destination data (must have the same size)
    std::vector<::bls12_381::scalar_t> dest_data(matrix_span.size());
    absl::Span<::bls12_381::scalar_t> dest_span(dest_data);
    // Apply from_montgomery and move data
    for (size_t i = 0; i < matrix_span.size(); ++i) {
      dest_span[i] =
          ::bls12_381::scalar_t::from_montgomery(reinterpret_cast_test[i]);
    }

    ::bls12_381::scalar_t* d_matrix;
    cudaMalloc(&d_matrix, current_matrix_size * sizeof(::bls12_381::scalar_t));
    cudaMemcpy(d_matrix, dest_span.data(),
               current_matrix_size * sizeof(::bls12_381::scalar_t),
               cudaMemcpyHostToDevice);

    leaves[idx] = {
        d_matrix,
        static_cast<size_t>(matrix.cols()),
        static_cast<size_t>(matrix.rows()),
    };
    ++idx;
  }

  /// Allocate memory for digests of {keep_rows} rows of a tree
  size_t digests_mem = digests_len * sizeof(::bls12_381::scalar_t);
  ::bls12_381::scalar_t* icicle_digest =
      static_cast<::bls12_381::scalar_t*>(malloc(digests_mem));

  cudaError_t error = tachyon_bls12_381_mmcs_commit_cuda(
      leaves, number_of_inputs, icicle_digest, &poseidon, &poseidon, *config_);

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
