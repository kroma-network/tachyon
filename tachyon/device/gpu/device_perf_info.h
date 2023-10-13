// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_DEVICE_GPU_DEVICE_PERF_INFO_H_
#define TACHYON_DEVICE_GPU_DEVICE_PERF_INFO_H_

#include <cstdint>
#include <optional>

#include "tachyon/build/build_config.h"
#include "tachyon/export.h"

namespace tachyon::device::gpu {

// These values are persistent to logs. Entries should not be renumbered and
// numeric values should never be reused.
enum class IntelGpuGeneration {
  kNonIntel = 0,
  kUnknownIntel = 1,  // Intel GPU, but not one of the following generations.
  kGen4 = 4,
  kGen5 = 5,
  kGen6 = 6,
  kGen7 = 7,
  kGen8 = 8,
  kGen9 = 9,
  kGen10 = 10,
  kGen11 = 11,
  kGen12 = 12,
  kMaxValue = kGen12,
};

struct TACHYON_EXPORT DevicePerfInfo {
  uint32_t total_physical_memory_mb = 0u;
  uint32_t total_disk_space_mb = 0u;
  uint32_t hardware_concurrency = 0u;

  IntelGpuGeneration intel_gpu_generation = IntelGpuGeneration::kNonIntel;
  bool software_rendering = false;
};

// Thread-safe getter and setter of global instance of DevicePerfInfo.
TACHYON_EXPORT std::optional<DevicePerfInfo> GetDevicePerfInfo();
TACHYON_EXPORT void SetDevicePerfInfo(const DevicePerfInfo& device_perf_info);

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_DEVICE_PERF_INFO_H_
