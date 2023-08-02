#ifndef TACHYON_DEVICE_GPU_SCOPED_MEMORY_H_
#define TACHYON_DEVICE_GPU_SCOPED_MEMORY_H_

#include <memory>
#include <ostream>
#include <string>

#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/export.h"

namespace tachyon::device::gpu {

enum class MemoryType {
  kDevice,
  kPageLocked,
#if TACHYON_CUDA
  kUnified,
#endif  // TACHYON_CUDA
};

TACHYON_EXPORT std::string MemoryTypeToString(MemoryType type);

TACHYON_EXPORT std::ostream& operator<<(std::ostream& os, MemoryType type);

template <MemoryType M>
struct MemoryDeleter {
  void operator()(void* dev_ptr) const {
    gpuError_t error = gpuSuccess;
    if constexpr (M == MemoryType::kDevice) {
      error = gpuFree(dev_ptr);
    } else if constexpr (M == MemoryType::kPageLocked) {
      error = gpuFreeHost(dev_ptr);
    }
#if TACHYON_CUDA
    else if constexpr (M == MemoryType::kUnified) {
      error = cudaFree(dev_ptr);
    }
#endif  // TACHYON_CUDA
    GPU_CHECK(error == gpuSuccess, error) << "Failed to free memory " << M;
  }
};

template <typename T, MemoryType M>
using ScopedMemory = std::unique_ptr<T, MemoryDeleter<M>>;

template <typename T>
ScopedMemory<T, MemoryType::kDevice> Malloc(size_t size) {
  T* ptr = nullptr;
  gpuError_t error = gpuMalloc(&ptr, sizeof(T) * size);
  GPU_CHECK(error == gpuSuccess, error) << "Failed to gpuMalloc()";
  return ScopedMemory<T, MemoryType::kDevice>(ptr);
}

template <typename T>
#if TACHYON_CUDA
ScopedMemory<T, MemoryType::kPageLocked> MallocHost(size_t size,
                                                    unsigned int flags) {
#elif TACHYON_USE_ROCM
ScopedMemory<T, MemoryType::kPageLocked> MallocHost(size_t size) {
#endif
  T* ptr = nullptr;
#if TACHYON_CUDA
  gpuError_t error = gpuMallocHost(&ptr, sizeof(T) * size, flags);
#elif TACHYON_USE_ROCM
  gpuError_t error = gpuMallocHost(&ptr, sizeof(T) * size);
#endif
  GPU_CHECK(error == gpuSuccess, error) << "Failed to gpuMallocHost()";
  return ScopedMemory<T, MemoryType::kPageLocked>(ptr);
}

template <typename T>
using ScopedDeviceMemory = ScopedMemory<T, MemoryType::kDevice>;
template <typename T>
using ScopedPageLockedMemory = ScopedMemory<T, MemoryType::kPageLocked>;
#if TACHYON_CUDA
template <typename T>
using ScopedUnifiedMemory = ScopedMemory<T, MemoryType::kUnified>;
#endif  // TACHYON_CUDA

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_SCOPED_MEMORY_H_
