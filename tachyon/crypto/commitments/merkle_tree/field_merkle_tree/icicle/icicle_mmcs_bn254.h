#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MMCS_BN254_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MMCS_BN254_H_

#include "third_party/icicle/include/curves/params/bn254.cu.h"
#include "third_party/icicle/include/hash/hash.cu.h"
#include "third_party/icicle/include/matrix/matrix.cu.h"
#include "third_party/icicle/include/merkle-tree/merkle_tree_config.h"

#include "tachyon/device/gpu/gpu_device_functions.h"

extern "C" gpuError_t tachyon_bn254_mmcs_commit_cuda(
    const ::matrix::Matrix<::bn254::scalar_t>* leaves,
    unsigned int number_of_inputs, ::bn254::scalar_t* digests,
    const ::hash::Hasher<::bn254::scalar_t, ::bn254::scalar_t>* hasher,
    const ::hash::Hasher<::bn254::scalar_t, ::bn254::scalar_t>* compressor,
    const ::merkle_tree::TreeBuilderConfig& tree_config);

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MMCS_BN254_H_
