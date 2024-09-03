#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MERKLE_TREE_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MERKLE_TREE_H_

#include "third_party/icicle/include/merkle-tree/merkle_tree_config.h"

#include "tachyon/base/logging.h"
#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/export.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::crypto {

template <class F>
struct IsIcicleMerkleTreeSupportedImpl {
  constexpr static bool value = false;
};

template <>
struct IsIcicleMerkleTreeSupportedImpl<math::BabyBear> {
  constexpr static bool value = true;
};

template <>
struct IsIcicleMerkleTreeSupportedImpl<math::bn254::Fr> {
  constexpr static bool value = true;
};

template <typename F>
constexpr bool IsIcicleMerkleTreeSupported =
    IsIcicleMerkleTreeSupportedImpl<F>::value;

struct TACHYON_EXPORT IcicleMerkleTreeOptions {
  unsigned int arity = 2;
  unsigned int keep_rows = 0;
  unsigned int digest_elements = 1;
  bool are_inputs_on_device = false;
  bool are_outputs_on_device = false;
  bool is_async = false;
};

template <typename F>
class IcicleMerkleTree {
 public:
  IcicleMerkleTree(
      gpuMemPool_t mem_pool, gpuStream_t stream,
      const IcicleMerkleTreeOptions& options = IcicleMerkleTreeOptions())
      : mem_pool_(mem_pool), stream_(stream) {
    ::device_context::DeviceContext ctx{stream_, /*device_id=*/
                                        0, mem_pool_};
    config_.reset(new ::merkle_tree::TreeBuilderConfig{
        ctx,
        options.arity,
        options.keep_rows,
        options.digest_elements,
        options.are_inputs_on_device,
        options.are_outputs_on_device,
        options.is_async,
    });
    VLOG(1) << "IcicleMerkleTree is created";
  }
  IcicleMerkleTree(const IcicleMerkleTree& other) = delete;
  IcicleMerkleTree& operator=(const IcicleMerkleTree& other) = delete;

  [[nodiscard]] bool Build(std::vector<math::RowMajorMatrix<F>>&& inputs,
                           F* digests, absl::Span<const F> round_constants,
                           absl::Span<const F> internal_matrix_diag);

 private:
  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;
  std::unique_ptr<::merkle_tree::TreeBuilderConfig> config_;
};

template <>
TACHYON_EXPORT bool IcicleMerkleTree<math::BabyBear>::Build(
    std::vector<math::RowMajorMatrix<math::BabyBear>>&& inputs,
    math::BabyBear* digests, absl::Span<const math::BabyBear> round_constants,
    absl::Span<const math::BabyBear> internal_matrix_diag);

template <>
TACHYON_EXPORT bool IcicleMerkleTree<math::bn254::Fr>::Build(
    std::vector<math::RowMajorMatrix<math::bn254::Fr>>&& inputs,
    math::bn254::Fr* digests, absl::Span<const math::bn254::Fr> round_constants,
    absl::Span<const math::bn254::Fr> internal_matrix_diag);

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FIELD_MERKLE_TREE_ICICLE_ICICLE_MERKLE_TREE_H_
