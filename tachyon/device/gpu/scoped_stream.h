#ifndef TACHYON_DEVICE_GPU_SCOPED_STREAM_H_
#define TACHYON_DEVICE_GPU_SCOPED_STREAM_H_

#include <memory>
#include <type_traits>

#include "tachyon/device/gpu/gpu_device_functions.h"
#include "tachyon/device/gpu/gpu_enums.h"
#include "tachyon/device/gpu/gpu_logging.h"
#include "tachyon/export.h"

namespace tachyon::device::gpu {

struct TACHYON_EXPORT StreamDestroyer {
  void operator()(gpuStream_t event) const {
    gpuError_t error = gpuStreamDestroy(event);
    GPU_CHECK(error == gpuSuccess, error) << "Failed to gpuStreamDestroy()";
  }
};

using ScopedStream =
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDestroyer>;

TACHYON_EXPORT ScopedStream CreateStream();
TACHYON_EXPORT ScopedStream CreateStreamWithFlags(unsigned int flags);

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_SCOPED_STREAM_H_
