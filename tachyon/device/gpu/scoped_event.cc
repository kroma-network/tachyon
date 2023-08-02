#include "tachyon/device/gpu/scoped_event.h"

namespace tachyon::device::gpu {

ScopedEvent CreateEvent() {
  gpuEvent_t event = nullptr;
  gpuError_t error = gpuEventCreate(&event);
  GPU_CHECK(error == gpuSuccess, error) << "Failed to gpuEventCreate()";
  return ScopedEvent(event);
}

ScopedEvent CreateEventWithFlags(unsigned int flags) {
  gpuEvent_t event = nullptr;
  gpuError_t error = gpuEventCreateWithFlags(&event, flags);
  GPU_CHECK(error == gpuSuccess, error)
      << "Failed to gpuEventCreateWithFlags()";
  return ScopedEvent(event);
}

}  // namespace tachyon::device::gpu
