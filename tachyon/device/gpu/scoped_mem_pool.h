#ifndef TACHYON_DEVICE_GPU_SCOPED_MEM_POOL_H_
#define TACHYON_DEVICE_GPU_SCOPED_MEM_POOL_H_

#include <memory>
#include <type_traits>

#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/export.h"

namespace tachyon::device::gpu {

struct TACHYON_EXPORT MemPoolDestroyer {
  void operator()(gpuMemPool_t mem_pool) const {
    gpuError_t error = gpuMemPoolDestroy(mem_pool);
    GPU_CHECK(error == gpuSuccess, error) << "Failed to gpuMemPoolDestroy()";
  }
};

using ScopedMemPool =
    std::unique_ptr<std::remove_pointer_t<gpuMemPool_t>, MemPoolDestroyer>;

TACHYON_EXPORT ScopedMemPool CreateMemPool(const gpuMemPoolProps* pool_props);

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_SCOPED_MEM_POOL_H_
