#ifndef TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FILED_MERKLE_TREE_ICICLE_ICICLE_MMCS_H_
#define TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FILED_MERKLE_TREE_ICICLE_ICICLE_MMCS_H_

#include <memory>

#include "absl/types/span.h"
#include "third_party/icicle/include/merkle-tree/merkle.cu.h"

#include "tachyon/base/logging.h"
#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/export.h"
#include "tachyon/math/elliptic_curves/bls12/bls12_381/fr.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/math/finite_fields/baby_bear/baby_bear.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::crypto {

template <class F>
struct IsIcicleMMCSSupportedImpl {
  constexpr static bool value = false;
};

template <>
struct IsIcicleMMCSSupportedImpl<math::BabyBear> {
  constexpr static bool value = true;
};

template <>
struct IsIcicleMMCSSupportedImpl<math::bls12_381::Fr> {
  constexpr static bool value = true;
};

template <>
struct IsIcicleMMCSSupportedImpl<math::bn254::Fr> {
  constexpr static bool value = true;
};

template <typename F>
constexpr bool IsIcicleMMCSSupported = IsIcicleMMCSSupportedImpl<F>::value;

struct TACHYON_EXPORT IcicleMMCSOptions {
  unsigned int arity = 2;
  unsigned int keep_rows = 0;
  unsigned int digest_elements = 1;
  bool are_inputs_on_device = false;
  bool are_outputs_on_device = false;
  bool is_async = false;
};

template <typename F>
class IcicleMMCS {
 public:
  IcicleMMCS(gpuMemPool_t mem_pool, gpuStream_t stream,
             const IcicleMMCSOptions& options = IcicleMMCSOptions())
      : mem_pool_(mem_pool), stream_(stream) {
    ::device_context::DeviceContext ctx{stream_, /*device_id=*/0, mem_pool_};
    config_.reset(new ::merkle_tree::TreeBuilderConfig{
        ctx,
        options.arity,
        options.keep_rows,
        options.digest_elements,
        options.are_inputs_on_device,
        options.are_outputs_on_device,
        options.is_async,
    });
    VLOG(1) << "IcicleMMCS is created";
  }
  IcicleMMCS(const IcicleMMCS& other) = delete;
  IcicleMMCS& operator=(const IcicleMMCS& other) = delete;

  [[nodiscard]] bool DoCommit(std::vector<math::RowMajorMatrix<F>>&& matrices);

 private:
  gpuMemPool_t mem_pool_ = nullptr;
  gpuStream_t stream_ = nullptr;
  std::unique_ptr<::merkle_tree::TreeBuilderConfig> config_;
};

template <>
TACHYON_EXPORT bool IcicleMMCS<math::bn254::Fr>::DoCommit(
    std::vector<math::RowMajorMatrix<math::bn254::Fr>>&& matrices);

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_MERKLE_TREE_FILED_MERKLE_TREE_ICICLE_ICICLE_MMCS_H_
