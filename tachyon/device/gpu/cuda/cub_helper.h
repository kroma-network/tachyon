#ifndef TACHYON_DEVICE_GPU_CUDA_CUB_HELPER_H_
#define TACHYON_DEVICE_GPU_CUDA_CUB_HELPER_H_

#include <tuple>

#include "tachyon/device/gpu/scoped_async_memory.h"
#include "tachyon/device/gpu/scoped_memory.h"

#define CUB_TRY_ALLOCATE(fn, ...)                                   \
  ({                                                                \
    size_t bytes = 0;                                               \
    cudaError_t error = fn(nullptr, bytes, __VA_ARGS__);            \
    if (error == cudaSuccess) {                                     \
      ::tachyon::device::gpu::ScopedDeviceMemory<uint8_t> storage = \
          ::tachyon::device::gpu::Malloc<uint8_t>(bytes);           \
      std::ignore = storage;                                        \
    } else {                                                        \
      GPU_LOG(ERROR, error) << "Failed to " #fn;                    \
    }                                                               \
    error;                                                          \
  })

#define CUB_INVOKE(fn, ...)                                         \
  ({                                                                \
    size_t bytes = 0;                                               \
    cudaError_t error = fn(nullptr, bytes, __VA_ARGS__);            \
    if (error == cudaSuccess) {                                     \
      ::tachyon::device::gpu::ScopedDeviceMemory<uint8_t> storage = \
          ::tachyon::device::gpu::Malloc<uint8_t>(bytes);           \
      error = fn(storage.get(), bytes, __VA_ARGS__);                \
    } else {                                                        \
      GPU_LOG(ERROR, error) << "Failed to " #fn;                    \
    }                                                               \
    error;                                                          \
  })

#define CUB_TRY_ALLOCATE_WITH_POOL(pool, stream, fn, ...)                   \
  ({                                                                        \
    size_t bytes = 0;                                                       \
    cudaError_t error = fn(nullptr, bytes, __VA_ARGS__);                    \
    if (error == cudaSuccess) {                                             \
      ::tachyon::device::gpu::ScopedAsyncMemory<uint8_t> storage =          \
          ::tachyon::device::gpu::MallocFromPoolAsync<uint8_t>(bytes, pool, \
                                                               stream);     \
      std::ignore = storage;                                                \
    } else {                                                                \
      GPU_LOG(ERROR, error) << "Failed to " #fn;                            \
    }                                                                       \
    error;                                                                  \
  })

#define CUB_INVOKE_WITH_POOL(pool, stream, fn, ...)                         \
  ({                                                                        \
    size_t bytes = 0;                                                       \
    cudaError_t error = fn(nullptr, bytes, __VA_ARGS__);                    \
    if (error == cudaSuccess) {                                             \
      ::tachyon::device::gpu::ScopedAsyncMemory<uint8_t> storage =          \
          ::tachyon::device::gpu::MallocFromPoolAsync<uint8_t>(bytes, pool, \
                                                               stream);     \
      error = fn(storage.get(), bytes, __VA_ARGS__);                        \
    } else {                                                                \
      GPU_LOG(ERROR, error) << "Failed to " #fn;                            \
    }                                                                       \
    error;                                                                  \
  })

#endif  // TACHYON_DEVICE_GPU_CUDA_CUB_HELPER_H_
