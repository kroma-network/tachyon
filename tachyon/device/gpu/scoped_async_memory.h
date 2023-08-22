#ifndef TACHYON_DEVICE_GPU_SCOPED_ASYNC_MEMORY_H_
#define TACHYON_DEVICE_GPU_SCOPED_ASYNC_MEMORY_H_

#include <utility>

#include "tachyon/device/gpu/gpu_logging.h"

namespace tachyon::device::gpu {

template <typename T>
class ScopedAsyncMemory {
 public:
  ScopedAsyncMemory() = default;
  ScopedAsyncMemory(T* ptr, gpuStream_t stream) : ptr_(ptr), stream_(stream) {}
  ScopedAsyncMemory(const ScopedAsyncMemory& other) = delete;
  ScopedAsyncMemory& operator=(const ScopedAsyncMemory& other) = delete;
  ScopedAsyncMemory(ScopedAsyncMemory&& other)
      : ptr_(std::exchange(other.ptr_, nullptr)),
        stream_(std::exchange(other.stream_, nullptr)) {}
  ScopedAsyncMemory& operator=(ScopedAsyncMemory&& other) {
    ptr_ = std::exchange(other.ptr_, nullptr);
    stream_ = std::exchange(other.stream_, nullptr);
    return *this;
  }
  ~ScopedAsyncMemory() { reset(); }

  T& operator*() { return *get(); }
  const T& operator*() const { return *get(); }

  T* operator->() { return get(); }
  const T* operator->() const { return get(); }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

  void reset() {
    if (ptr_) {
      GPU_MUST_SUCCESS(gpuFreeAsync(ptr_, stream_), "Failed to gpuFreeAsync()");
    }
    ptr_ = nullptr;
    stream_ = nullptr;
  }

 private:
  T* ptr_ = nullptr;
  gpuStream_t stream_ = nullptr;
};

template <typename T>
ScopedAsyncMemory<T> MallocFromPoolAsync(size_t size, gpuMemPool_t pool,
                                         gpuStream_t stream) {
  T* ptr = nullptr;
  GPU_MUST_SUCCESS(gpuMallocFromPoolAsync(&ptr, sizeof(T) * size, pool, stream),
                   "Failed to gpuMallocFromPoolAsync()");
  return {ptr, stream};
}

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_SCOPED_ASYNC_MEMORY_H_
