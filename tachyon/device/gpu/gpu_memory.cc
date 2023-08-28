#include "tachyon/device/gpu/gpu_memory.h"

#if TACHYON_CUDA
#define gpuMalloc cudaMalloc
#define gpuMallocHost cudaMallocHost
#define gpuMallocFromPoolAsync cudaMallocFromPoolAsync

#define gpuFree cudaFree
#define gpuFreeHost cudaFreeHost
#define gpuFreeAsync cudaFreeAsync

#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemset cudaMemset
#define gpuMemsetAsync cudaMemsetAsync

#define gpuPointerGetAttributes cudaPointerGetAttributes
#define gpuMemGetInfo cudaMemGetInfo
#else
#define gpuMalloc hipMalloc
#define gpuMallocHost hipMallocHost
#define gpuMallocFromPoolAsync hipMallocFromPoolAsync

#define gpuFree hipFree
#define gpuFreeHost hipFreeHost
#define gpuFreeAsync hipFreeAsync

#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemset hipMemset
#define gpuMemsetAsync hipMemsetAsync

#define gpuPointerGetAttributes hipPointerGetAttributes
#define gpuMemGetInfo hipMemGetInfo
#endif

namespace tachyon::device::gpu {

std::string GpuMemoryTypeToString(GpuMemoryType type) {
  switch (type) {
    case GpuMemoryType::kUnregistered:
      return "Unregistered";
    case GpuMemoryType::kDevice:
      return "Device";
    case GpuMemoryType::kHost:
      return "Host";
#if TACHYON_CUDA
    case GpuMemoryType::kUnified:
      return "Unified";
#endif  // TACHYON_CUDA
  }
  NOTREACHED();
  return "";
}

std::ostream& operator<<(std::ostream& os, GpuMemoryType type) {
  return os << GpuMemoryTypeToString(type);
}

gpuMemcpyKind ComputeGpuMemcpyKind(GpuMemoryType src_type,
                                   GpuMemoryType dst_type) {
  switch (src_type) {
    case GpuMemoryType::kUnregistered:
      NOTREACHED();
    case GpuMemoryType::kDevice:
      switch (dst_type) {
        case GpuMemoryType::kUnregistered:
          NOTREACHED();
        case GpuMemoryType::kDevice:
          return gpuMemcpyDeviceToDevice;
        case GpuMemoryType::kHost:
          return gpuMemcpyDeviceToHost;
        case GpuMemoryType::kUnified:
          return gpuMemcpyDefault;
      }
    case GpuMemoryType::kHost:
      switch (dst_type) {
        case GpuMemoryType::kUnregistered:
          NOTREACHED();
        case GpuMemoryType::kDevice:
          return gpuMemcpyHostToDevice;
        case GpuMemoryType::kHost:
          return gpuMemcpyHostToHost;
        case GpuMemoryType::kUnified:
          return gpuMemcpyDefault;
      }
    case GpuMemoryType::kUnified: {
      if (dst_type == GpuMemoryType::kUnregistered) {
        NOTREACHED();
      }
      return gpuMemcpyDefault;
    }
  }
  NOTREACHED();
  return gpuMemcpyDefault;
}

gpuError_t GpuMalloc(void** ptr, size_t size) { return gpuMalloc(ptr, size); }

gpuError_t GpuMallocHost(void** ptr, size_t size) {
  return gpuMallocHost(ptr, size);
}

#if TACHYON_CUDA
gpuError_t GpuMallocManaged(void** ptr, size_t size) {
  return cudaMallocManaged(ptr, size);
}
#endif  // TACHYON_CUDA

gpuError_t GpuMallocFromPoolAsync(void** ptr, size_t size, gpuMemPool_t pool,
                                  gpuStream_t stream) {
  return gpuMallocFromPoolAsync(ptr, size, pool, stream);
}

void GpuFreeMemory(gpuStream_t stream, void* ptr, GpuMemoryType type) {
  if (stream) {
    GPU_MUST_SUCCESS(gpuFreeAsync(ptr, stream), "Failed to gpuFreeAsync()");
  } else {
    switch (type) {
      case GpuMemoryType::kUnregistered:
        NOTREACHED();
      case GpuMemoryType::kDevice:
        GPU_MUST_SUCCESS(gpuFree(ptr), "Failed to gpuFree()");
        break;
      case GpuMemoryType::kHost:
        GPU_MUST_SUCCESS(gpuFreeHost(ptr), "Failed to gpuFreeHost()");
        break;
#if TACHYON_CUDA
      case GpuMemoryType::kUnified:
        GPU_MUST_SUCCESS(gpuFree(ptr), "Failed to gpuFree()");
        break;
#endif
    }
  }
}

gpuError_t GpuMemset(void* ptr, int value, size_t size) {
  return gpuMemset(ptr, value, size);
}

gpuError_t GpuMemsetAsync(void* ptr, int value, size_t size,
                          cudaStream_t stream) {
  return gpuMemsetAsync(ptr, value, size, stream);
}

gpuError_t GpuMemcpy(void* dst, const void* src, size_t size,
                     gpuMemcpyKind kind) {
  return gpuMemcpy(dst, src, size, kind);
}

gpuError_t GpuMemcpyAsync(void* dst, const void* src, size_t size,
                          gpuMemcpyKind kind, cudaStream_t stream) {
  return gpuMemcpyAsync(dst, src, size, kind, stream);
}

gpuError_t GpuPointerGetAttributes(gpuPointerAttributes* attributes,
                                   const void* ptr) {
  return gpuPointerGetAttributes(attributes, ptr);
}

gpuError_t GpuMemGetInfo(size_t* free, size_t* total) {
  return gpuMemGetInfo(free, total);
}

}  // namespace tachyon::device::gpu
