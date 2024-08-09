#include "tachyon/device/gpu/scoped_event.h"

namespace tachyon::device::gpu {

ScopedEvent CreateEvent() {
  gpuEvent_t event = nullptr;
  GPU_MUST_SUCCEED(gpuEventCreate(&event), "Failed gpuEventCreate()");
  return ScopedEvent(event);
}

ScopedEvent CreateEventWithFlags(unsigned int flags) {
  gpuEvent_t event = nullptr;
  GPU_MUST_SUCCEED(gpuEventCreateWithFlags(&event, flags),
                   "Failed gpuEventCreateWithFlags()");
  return ScopedEvent(event);
}

}  // namespace tachyon::device::gpu
