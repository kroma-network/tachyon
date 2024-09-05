#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs_baby_bear.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"
#include "third_party/icicle/include/poseidon2/poseidon2.cu.h"
#include "third_party/icicle/src/merkle-tree/merkle.cu.cc"  // NOLINT(build/include)
#include "third_party/icicle/src/merkle-tree/mmcs.cu.cc"  // NOLINT(build/include)

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/crypto/commitments/merkle_tree/field_merkle_tree/icicle/icicle_mmcs.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"

gpuError_t tachyon_babybear_mmcs_commit_cuda(
    const ::matrix::Matrix<::babybear::scalar_t>* leaves,
    unsigned int number_of_inputs, ::babybear::scalar_t* digests,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>* hasher,
    const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>*
        compressor,
    const ::merkle_tree::TreeBuilderConfig& tree_config) {
  // NOTE(GideokKim): The internal logic of the icicle Merkle tree always
  // assumes that the leaves exist in multiples of 2.
  return ::merkle_tree::mmcs_commit<::babybear::scalar_t, ::babybear::scalar_t>(
      leaves, number_of_inputs, digests, *hasher, *compressor, tree_config);
}

namespace tachyon::crypto {

template <>
bool IcicleMMCS<math::BabyBear>::Commit(
    std::vector<Eigen::Map<const math::RowMajorMatrix<math::BabyBear>>>&&
        matrices,
    std::vector<std::vector<std::vector<math::BabyBear>>>* outputs) {
#if FIELD_ID != BABY_BEAR
#error Only BABY_BEAR is supported
#endif
  std::vector<std::vector<::babybear::scalar_t>> matrices_tmp(matrices.size());
  for (size_t i = 0; i < matrices.size(); ++i) {
    matrices_tmp[i].resize(matrices[i].rows() * matrices[i].cols());
    OMP_PARALLEL_FOR(size_t j = 0; j < matrices[i].size(); ++j) {
      matrices_tmp[i][j] = ::babybear::scalar_t::from_montgomery(
          reinterpret_cast<const ::babybear::scalar_t*>(matrices[i].data())[j]);
    }
  }

  std::vector<::matrix::Matrix<::babybear::scalar_t>> leaves =
      base::CreateVector(matrices.size(), [&matrices_tmp, &matrices](size_t i) {
        return ::matrix::Matrix<::babybear::scalar_t>{
            matrices_tmp[i].data(),
            static_cast<size_t>(matrices[i].cols()),
            static_cast<size_t>(matrices[i].rows()),
        };
      });

  size_t max_tree_height = 0;
  for (const auto& matrix : matrices) {
    size_t tree_height = base::bits::Log2Ceiling(
        absl::bit_ceil(static_cast<size_t>(matrix.rows())));
    max_tree_height = std::max(max_tree_height, tree_height);
  }
  config_->keep_rows = max_tree_height + 1;
  config_->digest_elements = rate_;
  size_t digests_len = ::merkle_tree::get_digests_len(
      config_->keep_rows - 1, config_->arity, config_->digest_elements);

  std::unique_ptr<::babybear::scalar_t[]> icicle_digest(
      new ::babybear::scalar_t[digests_len]);

  gpuError_t error = tachyon_babybear_mmcs_commit_cuda(
      leaves.data(), leaves.size(), icicle_digest.get(),
      reinterpret_cast<
          const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>*>(
          hasher_),
      reinterpret_cast<
          const ::hash::Hasher<::babybear::scalar_t, ::babybear::scalar_t>*>(
          compressor_),
      *config_);
  if (error != gpuSuccess) {
    GPU_LOG(ERROR, error) << "Failed tachyon_babybear_mmcs_commit_cuda()";
    return false;
  }

  // TODO(GideokKim): Optimize this.
  outputs->reserve(config_->keep_rows);
  size_t previous_number_of_element = 0;
  for (size_t layer_idx = 0; layer_idx <= max_tree_height; ++layer_idx) {
    std::vector<std::vector<math::BabyBear>> digest_layer;
    size_t number_of_node = 1 << (max_tree_height - layer_idx);
    digest_layer.reserve(number_of_node);

    for (size_t node_idx = 0; node_idx < number_of_node; ++node_idx) {
      std::vector<math::BabyBear> digest;
      digest.reserve(config_->digest_elements);

      for (size_t element_idx = 0; element_idx < config_->digest_elements;
           ++element_idx) {
        size_t idx = previous_number_of_element +
                     config_->digest_elements * node_idx + element_idx;
        icicle_digest[idx] =
            ::babybear::scalar_t::to_montgomery(icicle_digest[idx]);
        digest.emplace_back(
            *reinterpret_cast<math::BabyBear*>(&icicle_digest[idx]));
      }
      digest_layer.emplace_back(std::move(digest));
    }
    outputs->emplace_back(std::move(digest_layer));
    previous_number_of_element += number_of_node * config_->digest_elements;
  }
  return true;
}

}  // namespace tachyon::crypto
