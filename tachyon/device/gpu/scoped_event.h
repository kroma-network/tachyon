#ifndef TACHYON_DEVICE_GPU_SCOPED_EVENT_H_
#define TACHYON_DEVICE_GPU_SCOPED_EVENT_H_

#include <memory>
#include <type_traits>

#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/export.h"

namespace tachyon::device::gpu {

struct TACHYON_EXPORT EventDestroyer {
  void operator()(gpuEvent_t event) const {
    GPU_MUST_SUCCEED(gpuEventDestroy(event), "Failed gpuEventDestroy()");
  }
};

using ScopedEvent =
    std::unique_ptr<std::remove_pointer_t<gpuEvent_t>, EventDestroyer>;

TACHYON_EXPORT ScopedEvent CreateEvent();
TACHYON_EXPORT ScopedEvent CreateEventWithFlags(unsigned int flags);

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_SCOPED_EVENT_H_
