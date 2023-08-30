#ifndef TACHYON_DEVICE_GPU_CUDA_CUB_HELPER_H_
#define TACHYON_DEVICE_GPU_CUDA_CUB_HELPER_H_

#include <tuple>

#include "tachyon/device/gpu/gpu_memory.h"

#define CUB_TRY_ALLOCATE(fn, ...)                                    \
  ({                                                                 \
    size_t bytes = 0;                                                \
    cudaError_t error = fn(nullptr, bytes, __VA_ARGS__);             \
    if (error == cudaSuccess) {                                      \
      ::tachyon::device::gpu::GpuMemory<uint8_t> storage =           \
          ::tachyon::device::gpu::GpuMemory<uint8_t>::Malloc(bytes); \
      std::ignore = storage;                                         \
    } else {                                                         \
      GPU_LOG(ERROR, error) << "Failed to " #fn;                     \
    }                                                                \
    error;                                                           \
  })

#define CUB_INVOKE(fn, ...)                                          \
  ({                                                                 \
    size_t bytes = 0;                                                \
    cudaError_t error = fn(nullptr, bytes, __VA_ARGS__);             \
    if (error == cudaSuccess) {                                      \
      ::tachyon::device::gpu::GpuMemory<uint8_t> storage =           \
          ::tachyon::device::gpu::GpuMemory<uint8_t>::Malloc(bytes); \
      error = fn(storage.get(), bytes, __VA_ARGS__);                 \
    } else {                                                         \
      GPU_LOG(ERROR, error) << "Failed to " #fn;                     \
    }                                                                \
    error;                                                           \
  })

#define CUB_TRY_ALLOCATE_WITH_POOL(pool, stream, fn, ...)                  \
  ({                                                                       \
    size_t bytes = 0;                                                      \
    cudaError_t error = fn(nullptr, bytes, __VA_ARGS__);                   \
    if (error == cudaSuccess) {                                            \
      ::tachyon::device::gpu::GpuMemory<uint8_t> storage =                 \
          ::tachyon::device::gpu::GpuMemory<uint8_t>::MallocFromPoolAsync( \
              bytes, pool, stream);                                        \
      std::ignore = storage;                                               \
    } else {                                                               \
      GPU_LOG(ERROR, error) << "Failed to " #fn;                           \
    }                                                                      \
    error;                                                                 \
  })

#define CUB_INVOKE_WITH_POOL(pool, stream, fn, ...)                        \
  ({                                                                       \
    size_t bytes = 0;                                                      \
    cudaError_t error = fn(nullptr, bytes, __VA_ARGS__);                   \
    if (error == cudaSuccess) {                                            \
      ::tachyon::device::gpu::GpuMemory<uint8_t> storage =                 \
          ::tachyon::device::gpu::GpuMemory<uint8_t>::MallocFromPoolAsync( \
              bytes, pool, stream);                                        \
      error = fn(storage.get(), bytes, __VA_ARGS__);                       \
    } else {                                                               \
      GPU_LOG(ERROR, error) << "Failed to " #fn;                           \
    }                                                                      \
    error;                                                                 \
  })

#endif  // TACHYON_DEVICE_GPU_CUDA_CUB_HELPER_H_
