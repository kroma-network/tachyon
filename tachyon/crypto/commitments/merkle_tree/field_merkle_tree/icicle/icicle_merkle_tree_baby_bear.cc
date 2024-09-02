#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_merkle_tree_baby_bear.h"

#include "absl/numeric/bits.h"
#include "third_party/icicle/include/fields/id.h"
#include "third_party/icicle/src/merkle-tree/merkle.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/poseidon2/constants.cu.cc"  // NOLINT(build/include)

#include "tachyon/base/bits.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_merkle_tree.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"

cudaError_t tachyon_babybear_build_merkle_tree(
    const ::babybear::scalar_t* leaves, ::babybear::scalar_t* digests,
    unsigned int height, unsigned int input_block_len,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>*
        compression,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>*
        bottom_layer,
    const ::merkle_tree::TreeBuilderConfig& tree_config) {
  return ::merkle_tree::build_merkle_tree<::babybear::scalar_t,
                                          ::babybear::scalar_t>(
      leaves, digests, height, input_block_len, *compression, *bottom_layer,
      tree_config);
}

namespace tachyon::crypto {

template <>
bool IcicleMerkleTree<math::BabyBear>::Build(
    std::vector<math::RowMajorMatrix<math::BabyBear>>&& inputs,
    math::BabyBear* digests) {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif

  size_t number_of_leaves = 0;
  size_t max_tree_height = 0;
  for (const auto& matrix : inputs) {
    size_t tree_height = base::bits::Log2Ceiling(
        absl::bit_ceil(static_cast<size_t>(matrix.size())));
    max_tree_height =
        max_tree_height > tree_height ? max_tree_height : tree_height;
    number_of_leaves += matrix.size();
  }

  ::babybear::scalar_t* leaves = static_cast<::babybear::scalar_t*>(
      malloc(number_of_leaves * sizeof(::babybear::scalar_t)));

  size_t idx = 0;
  for (const auto& matrix : inputs) {
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
      for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
        const auto* valuePtr = reinterpret_cast<const uint64_t*>(&matrix(i, j));
        leaves[idx] = ::babybear::scalar_t::from_montgomery(
            *reinterpret_cast<const ::babybear::scalar_t*>(valuePtr));
        ++idx;
      }
    }
  }

  // TODO(Noah)
  ::poseidon2::Poseidon2<::babybear::scalar_t> icicle_poseidon(
      16, 8, ::poseidon2::MdsType::DEFAULT_MDS,
      ::poseidon2::DiffusionStrategy::DEFAULT_DIFFUSION, config_->ctx);

  config_->keep_rows = 3;  // TODO(Noah)
  size_t digests_len = ::merkle_tree::get_digests_len(
      config_->keep_rows - 1, config_->arity, config_->digest_elements);

  ::babybear::scalar_t* ret = static_cast<::babybear::scalar_t*>(
      malloc(digests_len * sizeof(::babybear::scalar_t)));
  gpuError_t error = tachyon_babybear_build_merkle_tree(
      leaves, ret, max_tree_height, inputs.size(), &icicle_poseidon,
      &icicle_poseidon, *config_);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed tachyon_babybear_build_merkle_tree()";
    return false;
  }

  for (size_t idx = 0; idx < digests_len; ++idx) {
    *(ret + idx) = ::babybear::scalar_t::to_montgomery(*(ret + idx));
    // *(digests + idx) = *reinterpret_cast<math::BabyBear*>(ret + idx);
    LOG(ERROR) << *reinterpret_cast<math::BabyBear*>(ret + idx) << "\t";
  }

  free(leaves);
  free(ret);
  return true;
}

}  // namespace tachyon::crypto
