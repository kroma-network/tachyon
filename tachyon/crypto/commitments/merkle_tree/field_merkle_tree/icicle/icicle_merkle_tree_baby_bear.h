#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MERKLE_TREE_BABY_BEAR_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MERKLE_TREE_BABY_BEAR_H_

#include <stdint.h>

#include "third_party/icicle/include/fields/stark_fields/babybear.cu.h"
#include "third_party/icicle/include/merkle-tree/merkle.cu.h"
#include "third_party/icicle/include/poseidon2/poseidon2.cu.h"

extern "C" cudaError_t tachyon_babybear_build_merkle_tree(
    const ::babybear::scalar_t* leaves, ::babybear::scalar_t* digests,
    unsigned int height, unsigned int input_block_len,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>&
        compression,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>&
        bottom_layer,
    const ::merkle_tree::TreeBuilderConfig& tree_config);

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MERKLE_TREE_BABY_BEAR_H_
