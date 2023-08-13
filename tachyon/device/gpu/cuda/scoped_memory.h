#ifndef TACHYON_DEVICE_GPU_CUDA_SCOPED_MEMORY_H_
#define TACHYON_DEVICE_GPU_CUDA_SCOPED_MEMORY_H_

#include "tachyon/device/gpu/scoped_memory.h"

namespace tachyon::device::gpu {

template <typename T>
ScopedMemory<T, MemoryType::kUnified> MallocManaged(size_t size) {
  T* ptr = nullptr;
  GPU_MUST_SUCCESS(cudaMallocManaged(&ptr, sizeof(T) * size),
                   "Failed to cudaMallocManaged()");
  return ScopedMemory<T, MemoryType::kUnified>(ptr);
}

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_CUDA_SCOPED_MEMORY_H_
