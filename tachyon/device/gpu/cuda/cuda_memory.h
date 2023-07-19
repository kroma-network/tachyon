#ifndef TACHYON_DEVICE_GPU_CUDA_CUDA_MEMORY_H_
#define TACHYON_DEVICE_GPU_CUDA_CUDA_MEMORY_H_

#include "third_party/gpus/cuda/include/cuda_runtime.h"

#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/device/gpu/scoped_memory.h"

namespace tachyon {
namespace device {
namespace gpu {

template <typename T>
ScopedMemory<T> MakeManagedUnique(size_t size,
                                  unsigned int flags = cudaMemAttachGlobal) {
  T* ptr = nullptr;
  GPU_SUCCESS(cudaMallocManaged(&ptr, size, flags));
  ScopedMemory<T> ret;
  ret.reset(ptr);
  return ret;
}

}  // namespace gpu
}  // namespace device
}  // namespace tachyon

#endif  // TACHYON_DEVICE_GPU_CUDA_CUDA_MEMORY_H_
