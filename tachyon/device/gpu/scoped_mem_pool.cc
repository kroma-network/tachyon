#include "tachyon/device/gpu/scoped_mem_pool.h"

namespace tachyon::device::gpu {

ScopedMemPool CreateMemPool(const gpuMemPoolProps* pool_props) {
  gpuMemPool_t mem_pool = nullptr;
  gpuError_t error = gpuMemPoolCreate(&mem_pool, pool_props);
  GPU_CHECK(error == gpuSuccess, error) << "Failed to gpuMemPoolCreate()";
  return ScopedMemPool(mem_pool);
}

}  // namespace tachyon::device::gpu
