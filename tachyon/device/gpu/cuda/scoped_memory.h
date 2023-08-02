#ifndef TACHYON_DEVICE_GPU_CUDA_SCOPED_MEMORY_H_
#define TACHYON_DEVICE_GPU_CUDA_SCOPED_MEMORY_H_

#include "tachyon/device/gpu/scoped_memory.h"

namespace tachyon::device::gpu {

template <typename T>
ScopedMemory<T, MemoryType::kUnified> MallocManaged(size_t size) {
  T* ptr = nullptr;
  cudaError_t error = cudaMallocManaged(&ptr, sizeof(T) * size);
  GPU_CHECK(error == gpuSuccess, error);
  return ScopedMemory<T, MemoryType::kUnified>(ptr);
}

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_CUDA_SCOPED_MEMORY_H_
