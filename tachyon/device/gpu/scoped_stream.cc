#include "tachyon/device/gpu/scoped_stream.h"

namespace tachyon::device::gpu {

ScopedStream CreateStream() {
  gpuStream_t event = nullptr;
  GPU_MUST_SUCCESS(gpuStreamCreate(&event), "Failed gpuStreamCreate()");
  return ScopedStream(event);
}

ScopedStream CreateStreamWithFlags(unsigned int flags) {
  gpuStream_t event = nullptr;
  GPU_MUST_SUCCESS(gpuStreamCreateWithFlags(&event, flags),
                   "Failed gpuStreamCreateWithFlags()");
  return ScopedStream(event);
}

}  // namespace tachyon::device::gpu
