#ifndef TACHYON_DEVICE_GPU_GPU_DEVICE_MEMORY_H_
#define TACHYON_DEVICE_GPU_GPU_DEVICE_MEMORY_H_

#include "tachyon/device/gpu/gpu_device_functions.h"

namespace tachyon {
namespace device {
namespace gpu {

struct GpuDeviceMemoryDeleter {
  void operator()(void* devPtr) const { gpuFree(devPtr); }
};

using ScopedGpuDeviceMemory = std::unique_ptr<void, GpuDeviceMemoryDeleter>;

}  // namespace gpu
}  // namespace device
}  // namespace tachyon

#endif  // TACHYON_DEVICE_GPU_GPU_DEVICE_MEMORY_H_
