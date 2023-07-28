#include "tachyon/math/elliptic_curves/msm/kernels/msm_kernels.cu.h"

namespace tachyon::math::kernels {

namespace {

__global__ void InitializeBucketsKernelPfor(point_xyzz *buckets,
                                            const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  const auto bucket_index = gid / UINT4_COUNT;
  const auto element_index = gid % UINT4_COUNT;
  auto elements = reinterpret_cast<uint4 *>(&buckets[bucket_index].zz);
  memory::store<uint4, memory::st_modifier::cs>(elements + element_index, {});
}

}  // namespace

template <typename T>
bool set_kernel_attributes(T *func) {
  GPU_SUCCESS(cudaFuncSetCacheConfig(func, cudaFuncCachePreferL1));
  GPU_SUCCESS(
      cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout,
                           cudaSharedmemCarveoutMaxL1));
  return true;
}

bool SetMSMKernelAttributes() { GPU_SUCCESS() }

}  // namespace tachyon::math::kernels
