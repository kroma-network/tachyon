#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs_bls12_381.h"

#include "third_party/icicle/src/merkle-tree/merkle.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/merkle-tree/mmcs.cu.cc"  // NOLINT(build/include)

cudaError_t tachyon_bls12_381_build_merkle_tree(
    const ::bls12_381::scalar_t* leaves, ::bls12_381::scalar_t* digests,
    unsigned int height, unsigned int input_block_len,
    const ::hash::Hasher<::bls12_381::scalar_t, ::bls12_381::scalar_t>&
        compression,
    const ::hash::Hasher<::bls12_381::scalar_t, ::bls12_381::scalar_t>&
        bottom_layer,
    const ::merkle_tree::TreeBuilderConfig& tree_config) {
  return ::merkle_tree::build_merkle_tree<::bls12_381::scalar_t,
                                          ::bls12_381::scalar_t>(
      leaves, digests, height, input_block_len, compression, bottom_layer,
      tree_config);
}

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
