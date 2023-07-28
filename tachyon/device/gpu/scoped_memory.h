#ifndef TACHYON_DEVICE_GPU_SCOPED_MEMORY_H_
#define TACHYON_DEVICE_GPU_SCOPED_MEMORY_H_

#include <memory>

#include "tachyon/device/gpu/gpu_device_functions.h"

namespace tachyon::device::gpu {

struct MemoryDeleter {
  void operator()(void* dev_ptr) const { gpuFree(dev_ptr); }
};

template <typename T>
using ScopedMemory = std::unique_ptr<T, MemoryDeleter>;

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_SCOPED_MEMORY_H_
