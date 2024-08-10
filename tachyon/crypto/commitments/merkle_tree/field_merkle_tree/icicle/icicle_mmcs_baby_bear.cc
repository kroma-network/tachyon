#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs_baby_bear.h"

#include "third_party/icicle/src/merkle-tree/merkle.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/merkle-tree/mmcs.cu.cc"  // NOLINT(build/include)

cudaError_t tachyon_babybear_build_merkle_tree(
    const ::babybear::scalar_t* leaves, ::babybear::scalar_t* digests,
    unsigned int height, unsigned int input_block_len,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>&
        compression,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>&
        bottom_layer,
    const ::merkle_tree::TreeBuilderConfig& tree_config) {
  return ::merkle_tree::build_merkle_tree<::babybear::scalar_t,
                                          ::babybear::scalar_t>(
      leaves, digests, height, input_block_len, compression, bottom_layer,
      tree_config);
}

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
