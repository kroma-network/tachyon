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

bool MMCSCommitTest() {
#if FIELD_ID != BN254
#error Only BN254 is supported
#endif

  /// Tree of height N and arity A contains \sum{A^i} for i in 0..N elements
  uint32_t tree_arity = 2;
  uint32_t input_block_len = 600;
  uint32_t digest_elements = 8;
  uint32_t copied_matrices = 1;
  uint64_t tree_height = 3;
  uint64_t number_of_leaves = pow(tree_arity, tree_height);
  uint64_t total_number_of_leaves = number_of_leaves * input_block_len;

  ::device_context::DeviceContext ctx =
      ::device_context::get_default_device_context();

  uint32_t width = 16;
  uint32_t rate = 8;
  ::poseidon2::Poseidon2<::bn254::scalar_t> poseidon(
      width, rate, ::poseidon2::MdsType::PLONKY,
      ::poseidon2::DiffusionStrategy::MONTGOMERY, ctx);
  // ::poseidon::Poseidon<::bn254::scalar_t> poseidon(tree_arity, ctx);

  /// Use keep_rows to specify how many rows do you want to store
  int keep_rows = 3;
  size_t digests_len = ::merkle_tree::get_digests_len(keep_rows - 1, tree_arity,
                                                      digest_elements);

  ::bn254::scalar_t input = ::bn254::scalar_t::zero();

  // unsigned int number_of_inputs = tree_height * copied_matrices;
  unsigned int number_of_inputs = 1;
  ::matrix::Matrix<::bn254::scalar_t>* leaves =
      static_cast<::matrix::Matrix<::bn254::scalar_t>*>(malloc(
          number_of_inputs * sizeof(::matrix::Matrix<::bn254::scalar_t>)));
  uint64_t current_matrix_rows = number_of_leaves;

  for (int i = 0; i < number_of_inputs; i++) {
    uint64_t current_matrix_size = current_matrix_rows * input_block_len;
    for (int j = 0; j < copied_matrices; j++) {
      ::bn254::scalar_t* matrix = static_cast<::bn254::scalar_t*>(
          malloc(current_matrix_size * sizeof(::bn254::scalar_t)));

      for (uint64_t k = 0; k < current_matrix_size; k++) {
        matrix[k] = input;
        input = input + ::bn254::scalar_t::one();
      }

      leaves[i * copied_matrices + j] = {
          matrix,
          input_block_len,
          current_matrix_rows,
      };
    }

    current_matrix_rows /= tree_arity;
  }

  /// Allocate memory for digests of {keep_rows} rows of a tree
  size_t digests_mem = digests_len * sizeof(::bn254::scalar_t);
  ::bn254::scalar_t* icicle_digest =
      static_cast<::bn254::scalar_t*>(malloc(digests_mem));

  std::cout << "Number of leaves = " << number_of_leaves << std::endl;
  std::cout << "Total Number of leaves = " << total_number_of_leaves
            << std::endl;
  std::cout << "Memory for digests = " << digests_mem / 1024 / 1024 << " MB; "
            << digests_mem / 1024 / 1024 / 1024 << " GB" << std::endl;
  std::cout << "Number of digest elements = " << digests_len << std::endl;
  std::cout << std::endl;

  ::merkle_tree::TreeBuilderConfig tree_config =
      ::merkle_tree::default_merkle_config();
  tree_config.are_inputs_on_device = false;
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
