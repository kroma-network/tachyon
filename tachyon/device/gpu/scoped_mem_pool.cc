#include "tachyon/device/gpu/scoped_mem_pool.h"

namespace tachyon::device::gpu {

ScopedMemPool CreateMemPool(const gpuMemPoolProps* pool_props) {
  gpuMemPool_t mem_pool = nullptr;
  GPU_MUST_SUCCESS(gpuMemPoolCreate(&mem_pool, pool_props),
                   "Failed to gpuMemPoolCreate()");
  return ScopedMemPool(mem_pool);
}

}  // namespace tachyon::device::gpu
