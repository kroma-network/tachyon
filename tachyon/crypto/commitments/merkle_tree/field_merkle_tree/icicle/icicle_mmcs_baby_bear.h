#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MMCS_BABY_BEAR_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MMCS_BABY_BEAR_H_

#include "third_party/icicle/include/fields/stark_fields/babybear.cu.h"
#include "third_party/icicle/include/hash/hash.cu.h"
#include "third_party/icicle/include/matrix/matrix.cu.h"
#include "third_party/icicle/include/merkle-tree/merkle_tree_config.h"

#include "tachyon/device/gpu/gpu_device_functions.h"

extern "C" gpuError_t tachyon_babybear_mmcs_commit_cuda(
    const ::matrix::Matrix<::babybear::scalar_t>* leaves,
    unsigned int number_of_inputs, ::babybear::scalar_t* digests,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>* hasher,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>*
        compressor,
    const ::merkle_tree::TreeBuilderConfig& tree_config);

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MMCS_BABY_BEAR_H_
