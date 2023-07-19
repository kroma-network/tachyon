#ifndef TACHYON_DEVICE_GPU_GPU_LOGGING_H_
#define TACHYON_DEVICE_GPU_GPU_LOGGING_H_

#include "tachyon/base/logging.h"
#include "tachyon/device/gpu/gpu_device_functions.h"

#define GPU_SUCCESS(value) CHECK_EQ(value, cudaSuccess)
#define GPU_LOG_IF_ERROR(severity, error) \
  LOG_IF(severity, error != cudaSuccess) << cudaGetErrorString(error)

#endif  // TACHYON_DEVICE_GPU_GPU_LOGGING_H_
