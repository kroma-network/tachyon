#include "tachyon/device/gpu/scoped_event.h"

namespace tachyon::device::gpu {

ScopedEvent CreateEvent() {
  gpuEvent_t event = nullptr;
  GPU_MUST_SUCCESS(gpuEventCreate(&event), "Failed to gpuEventCreate()");
  return ScopedEvent(event);
}

ScopedEvent CreateEventWithFlags(unsigned int flags) {
  gpuEvent_t event = nullptr;
  GPU_MUST_SUCCESS(gpuEventCreateWithFlags(&event, flags),
                   "Failed to gpuEventCreateWithFlags()");
  return ScopedEvent(event);
}

}  // namespace tachyon::device::gpu
