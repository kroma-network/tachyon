#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FILED_MERKLE_TREE_ICICLE_ICICLE_MMCS_BLS12_381_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FILED_MERKLE_TREE_ICICLE_ICICLE_MMCS_BLS12_381_H_

#include <stdint.h>

#include "third_party/icicle/include/curves/params/bls12_381.cu.h"
#include "third_party/icicle/include/merkle-tree/merkle.cu.h"

extern "C" cudaError_t tachyon_bls12_381_build_merkle_tree(
    const ::bls12_381::scalar_t* leaves, ::bls12_381::scalar_t* digests,
    unsigned int height, unsigned int input_block_len,
    const ::hash::Hasher<::bls12_381::scalar_t, ::bls12_381::scalar_t>&
        compression,
    const ::hash::Hasher<::bls12_381::scalar_t, ::bls12_381::scalar_t>&
        bottom_layer,
    const ::merkle_tree::TreeBuilderConfig& tree_config);

extern "C" cudaError_t tachyon_bls12_381_mmcs_commit_cuda(
    const ::matrix::Matrix<::bls12_381::scalar_t>* leaves,
    unsigned int number_of_inputs, ::bls12_381::scalar_t* digests,
    const ::hash::Hasher<::bls12_381::scalar_t, ::bls12_381::scalar_t>* hasher,
    const ::hash::Hasher<::bls12_381::scalar_t, ::bls12_381::scalar_t>*
        compression,
    const ::merkle_tree::TreeBuilderConfig& tree_config);

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FILED_MERKLE_TREE_ICICLE_ICICLE_MMCS_BLS12_381_H_
