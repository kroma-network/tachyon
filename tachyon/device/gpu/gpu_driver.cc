#include "tachyon/device/gpu/gpu_driver.h"

#include "tachyon/base/logging.h"

namespace tachyon::device::gpu {

std::string MemorySpaceString(MemorySpace memory_space) {
  switch (memory_space) {
    case MemorySpace::kHost:
      return "host";
    case MemorySpace::kDevice:
      return "device";
    default:
      LOG(FATAL) << "impossible memory space";
  }
}

}  // namespace tachyon::device::gpu
