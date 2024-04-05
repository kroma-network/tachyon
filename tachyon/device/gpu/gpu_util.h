// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_DEVICE_GPU_GPU_UTIL_H_
#define TACHYON_DEVICE_GPU_GPU_UTIL_H_

#include <stdint.h>

#include <string>

#include "tachyon/build/build_config.h"
#include "tachyon/export.h"

namespace tachyon::device::gpu {

struct DevicePerfInfo;
struct GPUInfo;
struct GpuPreferences;
enum class IntelGpuSeriesType;
enum class IntelGpuGeneration;

TACHYON_EXPORT IntelGpuSeriesType GetIntelGpuSeriesType(uint32_t vendor_id,
                                                        uint32_t device_id);

TACHYON_EXPORT std::string GetIntelGpuGeneration(uint32_t vendor_id,
                                                 uint32_t device_id);

// If multiple Intel GPUs are detected, this returns the latest generation.
TACHYON_EXPORT IntelGpuGeneration
GetIntelGpuGeneration(const GPUInfo& gpu_info);

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_GPU_UTIL_H_
