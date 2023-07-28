#ifndef TACHYON_DEVICE_GPU_CUDA_CUDA_MEMORY_H_
#define TACHYON_DEVICE_GPU_CUDA_CUDA_MEMORY_H_

#include "third_party/gpus/cuda/include/cuda_runtime.h"

#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/device/gpu/scoped_memory.h"

namespace tachyon::device::gpu {

template <typename T>
ScopedMemory<T> MakeManagedUnique(size_t size,
                                  unsigned int flags = cudaMemAttachGlobal) {
  T* ptr = nullptr;
  cudaError_t error = cudaMallocManaged(&ptr, size, flags);
  GPU_CHECK(error == cudaSuccess, error);
  ScopedMemory<T> ret;
  ret.reset(ptr);
  return ret;
}

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_CUDA_CUDA_MEMORY_H_
