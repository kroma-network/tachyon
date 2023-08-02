#include "tachyon/device/gpu/scoped_stream.h"

namespace tachyon::device::gpu {

ScopedStream CreateStream() {
  gpuStream_t event = nullptr;
  gpuError_t error = gpuStreamCreate(&event);
  GPU_CHECK(error == gpuSuccess, error) << "Failed to gpuStreamCreate()";
  return ScopedStream(event);
}

ScopedStream CreateStreamWithFlags(unsigned int flags) {
  gpuStream_t event = nullptr;
  gpuError_t error = gpuStreamCreateWithFlags(&event, flags);
  GPU_CHECK(error == gpuSuccess, error)
      << "Failed to gpuStreamCreateWithFlags()";
  return ScopedStream(event);
}

}  // namespace tachyon::device::gpu
