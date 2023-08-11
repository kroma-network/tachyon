#ifndef TACHYON_DEVICE_GPU_GPU_DRIVER_H_
#define TACHYON_DEVICE_GPU_GPU_DRIVER_H_

#include <string>

#include "tachyon/device/gpu/gpu_types.h"
#include "tachyon/export.h"

namespace tachyon::device::gpu {

// Identifies the memory space where an allocation resides. See
// GpuDriver::GetPointerMemorySpace().
enum class MemorySpace { kHost, kDevice };

// Returns a casual string, such as "host" for the provided memory space.
TACHYON_EXPORT std::string MemorySpaceString(MemorySpace memory_space);

class GpuContext;

class TACHYON_EXPORT GpuDriver {};

// Ensures a context is activated within a scope.
class TACHYON_EXPORT ScopedActivateContext {
 public:
  // Activates the context via cuCtxSetCurrent, if it is not the currently
  // active context (a la cuCtxGetCurrent). Note the alternative push/pop
  // mechanism is said by NVIDIA to be relatively slow and deprecated.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7
  explicit ScopedActivateContext(GpuContext* context);

  // Checks that the context has remained activated for the duration of the
  // scope.
  ~ScopedActivateContext();

 private:
  GpuContext* to_restore_ = nullptr;
};

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_GPU_DRIVER_H_
