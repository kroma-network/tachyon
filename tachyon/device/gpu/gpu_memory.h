#ifndef TACHYON_DEVICE_GPU_MEMORY_H_
#define TACHYON_DEVICE_GPU_MEMORY_H_

#include <string>
#include <utility>

#include "tachyon/base/numerics/checked_math.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/export.h"

namespace tachyon::device::gpu {

enum class GpuMemoryType {
  kUnregistered,
  kDevice,
  kHost,
#if TACHYON_CUDA
  kUnified,
#endif  // TACHYON_CUDA
};

#if TACHYON_CUDA
using gpuMemoryType = cudaMemoryType;
#define gpuMemoryTypeUnregistered cudaMemoryTypeUnregistered
#define gpuMemoryTypeHost cudaMemoryTypeHost
#define gpuMemoryTypeDevice cudaMemoryTypeDevice
#define gpuMemoryTypeManaged cudaMemoryTypeManaged

using gpuMemcpyKind = cudaMemcpyKind;
#define gpuMemcpyHostToHost cudaMemcpyHostToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDefault cudaMemcpyDefault

using gpuPointerAttributes = cudaPointerAttributes;

#else
using gpuMemoryType = hipMemoryType;
#define gpuMemoryTypeUnregistered hipMemoryTypeUnregistered
#define gpuMemoryTypeHost hipMemoryTypeHost
#define gpuMemoryTypeDevice hipMemoryTypeDevice
#define gpuMemoryTypeManaged hipMemoryTypeManaged

using gpuMemcpyKind = hipMemcpyKind;
#define gpuMemcpyHostToHost hipMemcpyHostToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDefault hipMemcpyDefault

using gpuPointerAttributes = hipPointerAttribute_t;
#endif

TACHYON_EXPORT std::string GpuMemoryTypeToString(GpuMemoryType type);

TACHYON_EXPORT std::ostream& operator<<(std::ostream& os, GpuMemoryType type);

TACHYON_EXPORT gpuMemcpyKind ComputeGpuMemcpyKind(GpuMemoryType src_type,
                                                  GpuMemoryType dst_type);

TACHYON_EXPORT gpuError_t GpuMalloc(void** ptr, size_t size);
TACHYON_EXPORT gpuError_t GpuMallocHost(void** ptr, size_t size);
#if TACHYON_CUDA
TACHYON_EXPORT gpuError_t GpuMallocManaged(void** ptr, size_t size);
#endif  // TACHYON_CUDA
TACHYON_EXPORT gpuError_t GpuMallocFromPoolAsync(void** ptr, size_t size,
                                                 gpuMemPool_t pool,
                                                 gpuStream_t stream = nullptr);

TACHYON_EXPORT void GpuFreeMemory(gpuStream_t stream, void* ptr,
                                  GpuMemoryType type);

TACHYON_EXPORT gpuError_t GpuMemset(void* ptr, int value, size_t size);
TACHYON_EXPORT gpuError_t GpuMemsetAsync(void* ptr, int value, size_t size,
                                         gpuStream_t stream = nullptr);

TACHYON_EXPORT gpuError_t GpuMemcpy(void* dst, const void* src, size_t size,
                                    gpuMemcpyKind kind);
TACHYON_EXPORT gpuError_t GpuMemcpyAsync(void* dst, const void* src,
                                         size_t size, gpuMemcpyKind kind,
                                         gpuStream_t stream = nullptr);

TACHYON_EXPORT gpuError_t
GpuPointerGetAttributes(gpuPointerAttributes* attributes, const void* ptr);

TACHYON_EXPORT gpuError_t GpuMemGetInfo(size_t* free, size_t* total);

TACHYON_EXPORT

template <typename T>
class GpuMemory {
 public:
  static GpuMemory Malloc(size_t size) {
    T* ptr = nullptr;
    GPU_MUST_SUCCESS(
        GpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * size),
        "Failed to GpuMalloc()");
    return GpuMemory(ptr, size, GpuMemoryType::kDevice);
  }

  static GpuMemory MallocHost(size_t size) {
    T* ptr = nullptr;
    GPU_MUST_SUCCESS(
        GpuMallocHost(reinterpret_cast<void**>(&ptr), sizeof(T) * size),
        "Failed to GpuMallocHost()");
    return GpuMemory(ptr, size, GpuMemoryType::kHost);
  }

#if TACHYON_CUDA
  static GpuMemory MallocManaged(size_t size) {
    T* ptr = nullptr;
    GPU_MUST_SUCCESS(cudaMallocManaged(&ptr, sizeof(T) * size),
                     "Failed to cudaMallocManaged()");
    return GpuMemory(ptr, size, GpuMemoryType::kUnified);
  }
#endif  // TACHYON_CUDA

  static GpuMemory MallocFromPoolAsync(size_t size, gpuMemPool_t pool,
                                       gpuStream_t stream) {
    T* ptr = nullptr;
    GPU_MUST_SUCCESS(GpuMallocFromPoolAsync(reinterpret_cast<void**>(&ptr),
                                            sizeof(T) * size, pool, stream),
                     "Failed to GpuMallocFromPoolAsync()");
    return GpuMemory(ptr, size, GpuMemoryType::kDevice, stream);
  }

  static GpuMemory FromRaw(T* ptr, size_t size, GpuMemoryType memory_type,
                           gpuStream_t stream = nullptr) {
    return GpuMemory(ptr, size, memory_type, stream);
  }

  GpuMemory() = default;
  GpuMemory(const GpuMemory& other) = delete;
  GpuMemory& operator=(const GpuMemory& other) = delete;
  GpuMemory(GpuMemory&& other)
      : ptr_(std::exchange(other.ptr_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        memory_type_(
            std::exchange(other.memory_type_, GpuMemoryType::kUnregistered)),
        stream_(std::exchange(other.stream_, nullptr)) {}
  GpuMemory& operator=(GpuMemory&& other) {
    ptr_ = std::exchange(other.ptr_, nullptr);
    size_ = std::exchange(other.size_, 0);
    memory_type_ =
        std::exchange(other.memory_type_, GpuMemoryType::kUnregistered);
    stream_ = std::exchange(other.stream_, nullptr);
    return *this;
  }
  ~GpuMemory() { reset(); }

  size_t size() const { return size_; }
  gpuStream_t stream() const { return stream_; }
  GpuMemoryType memory_type() const { return memory_type_; }

  T& operator*() { return *get(); }
  const T& operator*() const { return *get(); }

  T* operator->() { return get(); }
  const T* operator->() const { return get(); }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

  T& operator[](size_t idx) { return ptr_[idx]; }
  const T& operator[](size_t idx) const { return ptr_[idx]; }

  void reset() {
    if (ptr_) {
      GpuFreeMemory(stream_, ptr_, memory_type_);
    }
    ptr_ = nullptr;
    size_ = 0;
    stream_ = nullptr;
  }

  bool GetAttributes(gpuPointerAttributes* attributes) const {
    gpuError_t error = GpuPointerGetAttributes(attributes, ptr_);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to GpuPointerGetAttributes()";
      return false;
    }
    return true;
  }

  bool Memset(int value = 0, size_t from = 0, size_t len = 0) {
    if (len == 0) {
      len = size_;
    }
    base::CheckedNumeric<size_t> checked_from = from;
    CHECK((checked_from + len).IsValid());
    gpuError_t error = GpuMemset(ptr_ + from, value, sizeof(T) * len);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to GpuMemset()";
      return false;
    }
    return true;
  }

  bool MemsetAsync(int value, gpuStream_t stream = nullptr, size_t from = 0,
                   size_t len = 0) {
    if (len == 0) {
      len = size_;
    }
    base::CheckedNumeric<size_t> checked_from = from;
    CHECK((checked_from + len).IsValid());
    gpuError_t error =
        GpuMemsetAsync(ptr_ + from, value, sizeof(T) * len, stream);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to GpuMemsetAsync()";
      return false;
    }
    return true;
  }

  bool CopyTo(void* dst, GpuMemoryType dst_memory_type, size_t from = 0,
              size_t len = 0) const {
    if (len == 0) {
      len = size_;
    }
    base::CheckedNumeric<size_t> checked_from = from;
    CHECK((checked_from + len).IsValid());
    gpuError_t error =
        GpuMemcpy(dst, ptr_ + from, sizeof(T) * len,
                  ComputeGpuMemcpyKind(memory_type_, dst_memory_type));
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to GpuMemcpy()";
      return false;
    }
    return true;
  }

  template <typename U>
  bool CopyTo(const GpuMemory<U>& dst_memory, size_t from = 0, size_t len = 0) {
    return CopyTo(dst_memory.get(), dst_memory.memory_type(), from, len);
  }

  bool CopyToAsync(void* dst, GpuMemoryType dst_memory_type,
                   gpuStream_t stream = nullptr, size_t from = 0,
                   size_t len = 0) const {
    if (len == 0) {
      len = size_;
    }
    base::CheckedNumeric<size_t> checked_from = from;
    CHECK((checked_from + len).IsValid());
    gpuError_t error = GpuMemcpyAsync(
        dst, ptr_ + from, sizeof(T) * len,
        ComputeGpuMemcpyKind(memory_type_, dst_memory_type), stream);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to GpuMemcpyAsync()";
      return false;
    }
    return true;
  }

  template <typename U>
  bool CopyToAsync(const GpuMemory<U>& dst_memory, gpuStream_t stream = nullptr,
                   size_t from = 0, size_t len = 0) {
    return CopyToAsync(dst_memory.get(), dst_memory.memory_type(), stream, from,
                       len);
  }

  bool CopyFrom(const void* src, GpuMemoryType src_memory_type, size_t from = 0,
                size_t len = 0) {
    if (len == 0) {
      len = size_;
    }
    base::CheckedNumeric<size_t> checked_from = from;
    CHECK((checked_from + len).IsValid());
    gpuError_t error =
        GpuMemcpy(ptr_ + from, src, sizeof(T) * len,
                  ComputeGpuMemcpyKind(src_memory_type, memory_type_));
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to GpuMemcpy()";
      return false;
    }
    return true;
  }

  template <typename U>
  bool CopyFrom(const GpuMemory<U>& src_memory, size_t from = 0,
                size_t len = 0) {
    return CopyFrom(src_memory.get(), src_memory.memory_type(), from, len);
  }

  bool CopyFromAsync(const void* src, GpuMemoryType src_memory_type,
                     gpuStream_t stream = nullptr, size_t from = 0,
                     size_t len = 0) {
    if (len == 0) {
      len = size_;
    }
    base::CheckedNumeric<size_t> checked_from = from;
    CHECK((checked_from + len).IsValid());
    gpuError_t error = GpuMemcpyAsync(
        ptr_ + from, src, sizeof(T) * len,
        ComputeGpuMemcpyKind(src_memory_type, memory_type_), stream);
    if (error != gpuSuccess) {
      GPU_LOG(ERROR, error) << "Failed to GpuMemcpyAsync()";
      return false;
    }
    return true;
  }

  template <typename U>
  bool CopyFromAsync(const GpuMemory<U>& src_memory,
                     gpuStream_t stream = nullptr, size_t from = 0,
                     size_t len = 0) {
    return CopyFromAsync(src_memory.get(), src_memory.memory_type(), stream,
                         from, len);
  }

  template <typename R>
  bool ToStdVector(std::vector<R>* ret) const {
    ret->resize(size_);
    return CopyTo(ret->data(), GpuMemoryType::kHost);
  }

  template <typename R>
  bool ToStdVectorAsync(std::vector<R>* ret,
                        gpuStream_t stream = nullptr) const {
    ret->resize(size_);
    return CopyToAsync(ret->data(), GpuMemoryType::kHost, stream);
  }

 private:
  GpuMemory(T* ptr, size_t size, GpuMemoryType memory_type,
            gpuStream_t stream = nullptr)
      : ptr_(ptr), size_(size), memory_type_(memory_type), stream_(stream) {}

  T* ptr_ = nullptr;
  size_t size_ = 0;
  GpuMemoryType memory_type_ = GpuMemoryType::kUnregistered;
  gpuStream_t stream_ = nullptr;
};

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_MEMORY_H_
