#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FILED_MERKLE_TREE_ICICLE_ICICLE_MMCS_BN254_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FILED_MERKLE_TREE_ICICLE_ICICLE_MMCS_BN254_H_

#include <stdint.h>

#include "third_party/icicle/include/curves/params/bn254.cu.h"
#include "third_party/icicle/include/merkle-tree/merkle.cu.h"
#include "third_party/icicle/include/poseidon/constants.cu.h"
#include "third_party/icicle/include/poseidon/poseidon.cu.h"
#include "third_party/icicle/include/poseidon2/constants.cu.h"
#include "third_party/icicle/include/poseidon2/poseidon2.cu.h"

extern "C" cudaError_t tachyon_bn254_mmcs_commit_cuda(
    const ::matrix::Matrix<::bn254::scalar_t>* leaves,
    unsigned int number_of_inputs, ::bn254::scalar_t* digests,
    const ::hash::Hasher<::bn254::scalar_t, ::bn254::scalar_t>* hasher,
    const ::hash::Hasher<::bn254::scalar_t, ::bn254::scalar_t>* compression,
    const ::merkle_tree::TreeBuilderConfig& tree_config);

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FILED_MERKLE_TREE_ICICLE_ICICLE_MMCS_BN254_H_
