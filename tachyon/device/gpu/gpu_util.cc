// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/device/gpu/gpu_util.h"

#include <vector>

#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/device/gpu/device_perf_info.h"
#include "tachyon/device/gpu/gpu_info.h"

namespace tachyon::device::gpu {

IntelGpuSeriesType GetIntelGpuSeriesType(uint32_t vendor_id,
                                         uint32_t device_id) {
  // Note that this function's output should only depend on vendor_id and
  // device_id of a GPU. This is because we record a histogram on the output
  // and we don't want to expose an extra bit other than the already recorded
  // vendor_id and device_id.
  if (vendor_id == 0x8086) {  // Intel
    // The device IDs of Intel GPU are based on following Mesa source files:
    // include/pci_ids/i965_pci_ids.h and iris_pci_ids.h
    uint32_t masked_device_id = device_id & 0xFF00;
    switch (masked_device_id) {
      case 0x2900:
        return IntelGpuSeriesType::kBroadwater;
      case 0x2A00:
        if (device_id == 0x2A02 || device_id == 0x2A12)
          return IntelGpuSeriesType::kBroadwater;
        if (device_id == 0x2A42) return IntelGpuSeriesType::kEaglelake;
        break;
      case 0x2E00:
        return IntelGpuSeriesType::kEaglelake;
      case 0x0000:
        return IntelGpuSeriesType::kIronlake;
      case 0x0100:
        if (device_id == 0x0152 || device_id == 0x0156 || device_id == 0x015A ||
            device_id == 0x0162 || device_id == 0x0166 || device_id == 0x016A)
          return IntelGpuSeriesType::kIvybridge;
        if (device_id == 0x0155 || device_id == 0x0157)
          return IntelGpuSeriesType::kBaytrail;
        return IntelGpuSeriesType::kSandybridge;
      case 0x0F00:
        return IntelGpuSeriesType::kBaytrail;
      case 0x0A00:
        if (device_id == 0x0A84) return IntelGpuSeriesType::kApollolake;
        return IntelGpuSeriesType::kHaswell;
      case 0x0400:
      case 0x0C00:
      case 0x0D00:
        return IntelGpuSeriesType::kHaswell;
      case 0x2200:
        return IntelGpuSeriesType::kCherrytrail;
      case 0x1600:
        return IntelGpuSeriesType::kBroadwell;
      case 0x5A00:
        if (device_id == 0x5A85 || device_id == 0x5A84)
          return IntelGpuSeriesType::kApollolake;
        return IntelGpuSeriesType::kCannonlake;
      case 0x1900:
        return IntelGpuSeriesType::kSkylake;
      case 0x1A00:
        return IntelGpuSeriesType::kApollolake;
      case 0x3100:
        return IntelGpuSeriesType::kGeminilake;
      case 0x5900:
        if (device_id == 0x591C) return IntelGpuSeriesType::kAmberlake;
        return IntelGpuSeriesType::kKabylake;
      case 0x8700:
        if (device_id == 0x87C0) return IntelGpuSeriesType::kKabylake;
        if (device_id == 0x87CA) return IntelGpuSeriesType::kCoffeelake;
        break;
      case 0x3E00:
        if (device_id == 0x3EA0 || device_id == 0x3EA1 || device_id == 0x3EA2 ||
            device_id == 0x3EA4 || device_id == 0x3EA3)
          return IntelGpuSeriesType::kWhiskeylake;
        return IntelGpuSeriesType::kCoffeelake;
      case 0x9B00:
        return IntelGpuSeriesType::kCometlake;
      case 0x8A00:
        return IntelGpuSeriesType::kIcelake;
      case 0x4500:
        return IntelGpuSeriesType::kElkhartlake;
      case 0x4E00:
        return IntelGpuSeriesType::kJasperlake;
      case 0x9A00:
        return IntelGpuSeriesType::kTigerlake;
      case 0x4c00:
        return IntelGpuSeriesType::kRocketlake;
      case 0x4900:
        return IntelGpuSeriesType::kDG1;
      case 0x4600:
        return IntelGpuSeriesType::kAlderlake;
      case 0x4F00:
      case 0x5600:
        return IntelGpuSeriesType::kAlchemist;
      case 0xa700:
        return IntelGpuSeriesType::kRaptorlake;
      case 0x7d00:
        return IntelGpuSeriesType::kMeteorlake;
      default:
        break;
    }
  }
  return IntelGpuSeriesType::kUnknown;
}

std::string GetIntelGpuGeneration(uint32_t vendor_id, uint32_t device_id) {
  if (vendor_id == 0x8086) {
    IntelGpuSeriesType gpu_series = GetIntelGpuSeriesType(vendor_id, device_id);
    switch (gpu_series) {
      case IntelGpuSeriesType::kBroadwater:
      case IntelGpuSeriesType::kEaglelake:
        return "4";
      case IntelGpuSeriesType::kIronlake:
        return "5";
      case IntelGpuSeriesType::kSandybridge:
        return "6";
      case IntelGpuSeriesType::kBaytrail:
      case IntelGpuSeriesType::kIvybridge:
      case IntelGpuSeriesType::kHaswell:
        return "7";
      case IntelGpuSeriesType::kCherrytrail:
      case IntelGpuSeriesType::kBroadwell:
        return "8";
      case IntelGpuSeriesType::kApollolake:
      case IntelGpuSeriesType::kSkylake:
      case IntelGpuSeriesType::kGeminilake:
      case IntelGpuSeriesType::kKabylake:
      case IntelGpuSeriesType::kAmberlake:
      case IntelGpuSeriesType::kCoffeelake:
      case IntelGpuSeriesType::kWhiskeylake:
      case IntelGpuSeriesType::kCometlake:
        return "9";
      case IntelGpuSeriesType::kCannonlake:
        return "10";
      case IntelGpuSeriesType::kIcelake:
      case IntelGpuSeriesType::kElkhartlake:
      case IntelGpuSeriesType::kJasperlake:
        return "11";
      case IntelGpuSeriesType::kTigerlake:
      case IntelGpuSeriesType::kRocketlake:
      case IntelGpuSeriesType::kDG1:
      case IntelGpuSeriesType::kAlderlake:
      case IntelGpuSeriesType::kAlchemist:
      case IntelGpuSeriesType::kRaptorlake:
      case IntelGpuSeriesType::kMeteorlake:
        return "12";
      default:
        break;
    }
  }
  return "";
}

IntelGpuGeneration GetIntelGpuGeneration(const GPUInfo& gpu_info) {
  const uint32_t kIntelVendorId = 0x8086;
  IntelGpuGeneration latest = IntelGpuGeneration::kNonIntel;
  std::vector<uint32_t> intel_device_ids;
  if (gpu_info.gpu.vendor_id == kIntelVendorId)
    intel_device_ids.push_back(gpu_info.gpu.device_id);
  for (const auto& gpu : gpu_info.secondary_gpus) {
    if (gpu.vendor_id == kIntelVendorId)
      intel_device_ids.push_back(gpu.device_id);
  }
  if (intel_device_ids.empty()) return latest;
  latest = IntelGpuGeneration::kUnknownIntel;
  for (uint32_t device_id : intel_device_ids) {
    std::string gen_str = gpu::GetIntelGpuGeneration(kIntelVendorId, device_id);
    int gen_int = 0;
    if (gen_str.empty() || !base::StringToInt(gen_str, &gen_int)) continue;
    DCHECK_GE(gen_int, static_cast<int>(IntelGpuGeneration::kUnknownIntel));
    DCHECK_LE(gen_int, static_cast<int>(IntelGpuGeneration::kMaxValue));
    if (gen_int > static_cast<int>(latest))
      latest = static_cast<IntelGpuGeneration>(gen_int);
  }
  return latest;
}

}  // namespace tachyon::device::gpu
