#ifndef TACHYON_DEVICE_GPU_GPU_LOGGING_H_
#define TACHYON_DEVICE_GPU_GPU_LOGGING_H_

#include "tachyon/base/logging.h"
#include "tachyon/device/gpu/gpu_device_functions.h"

#define GPU_SUCCESS(value) CHECK_EQ(value, cudaSuccess)

#endif  // TACHYON_DEVICE_GPU_GPU_LOGGING_H_
