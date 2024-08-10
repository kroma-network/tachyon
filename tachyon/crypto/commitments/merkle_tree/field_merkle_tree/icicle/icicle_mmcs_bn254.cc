#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs_bn254.h"

#include "third_party/icicle/src/merkle-tree/merkle.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/merkle-tree/mmcs.cu.cc"  // NOLINT(build/include)

cudaError_t tachyon_bn254_build_merkle_tree(
    const ::bn254::scalar_t* leaves, ::bn254::scalar_t* digests,
    unsigned int height, unsigned int input_block_len,
    const ::hash::Hasher<::bn254::scalar_t, ::bn254::scalar_t>& compression,
    const ::hash::Hasher<::bn254::scalar_t, ::bn254::scalar_t>& bottom_layer,
    const ::merkle_tree::TreeBuilderConfig& tree_config) {
  return ::merkle_tree::build_merkle_tree<::bn254::scalar_t, ::bn254::scalar_t>(
      leaves, digests, height, input_block_len, compression, bottom_layer,
      tree_config);
}

cudaError_t tachyon_bn254_mmcs_commit_cuda(
    const ::matrix::Matrix<::bn254::scalar_t>* leaves,
    unsigned int number_of_inputs, ::bn254::scalar_t* digests,
    const ::hash::Hasher<::bn254::scalar_t, ::bn254::scalar_t>* hasher,
    const ::hash::Hasher<::bn254::scalar_t, ::bn254::scalar_t>* compression,
    const ::merkle_tree::TreeBuilderConfig& tree_config) {
  return ::merkle_tree::mmcs_commit<::bn254::scalar_t, ::bn254::scalar_t>(
      leaves, number_of_inputs, digests, *hasher, *compression, tree_config);
}
