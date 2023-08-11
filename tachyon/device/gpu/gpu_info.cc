// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/device/gpu/gpu_info.h"

#include <stdint.h>

#include "tachyon/base/logging.h"
#include "tachyon/build/build_config.h"
#include "tachyon/device/gpu/gpu_util.h"

namespace tachyon::device::gpu {

namespace {

void EnumerateGPUDevice(const GPUInfo::GPUDevice& device,
                        GPUInfo::Enumerator* enumerator) {
  enumerator->BeginGPUDevice();
  enumerator->AddInt("vendorId", device.vendor_id);
  enumerator->AddInt("deviceId", device.device_id);
#if BUILDFLAG(IS_WIN) || BUILDFLAG(IS_CHROMEOS)
  enumerator->AddInt("revision", device.revision);
#endif
#if BUILDFLAG(IS_WIN)
  enumerator->AddInt("subSysId", device.sub_sys_id);
#endif  // BUILDFLAG(IS_WIN)
  enumerator->AddBool("active", device.active);
  enumerator->AddString("vendorString", device.vendor_string);
  enumerator->AddString("deviceString", device.device_string);
  enumerator->AddString("driverVendor", device.driver_vendor);
  enumerator->AddString("driverVersion", device.driver_version);
  enumerator->AddInt("cudaComputeCapabilityMajor",
                     device.cuda_compute_capability_major);
  enumerator->EndGPUDevice();
}

}  // namespace

GPUInfo::GPUDevice::GPUDevice() = default;

GPUInfo::GPUDevice::GPUDevice(const GPUInfo::GPUDevice& other) = default;

GPUInfo::GPUDevice::GPUDevice(GPUInfo::GPUDevice&& other) noexcept = default;

GPUInfo::GPUDevice::~GPUDevice() noexcept = default;

GPUInfo::GPUDevice& GPUInfo::GPUDevice::operator=(
    const GPUInfo::GPUDevice& other) = default;

GPUInfo::GPUDevice& GPUInfo::GPUDevice::operator=(
    GPUInfo::GPUDevice&& other) noexcept = default;

bool GPUInfo::GPUDevice::IsSoftwareRenderer() const {
  switch (vendor_id) {
    case 0x0000:  // Info collection failed to identify a GPU
    case 0xffff:  // Chromium internal flag for software rendering
    case 0x15ad:  // VMware
    case 0x1414:  // Microsoft software renderer
      return true;
    default:
      return false;
  }
}

GPUInfo::GPUInfo() : optimus(false), amd_switchable(false) {}

GPUInfo::GPUInfo(const GPUInfo& other) = default;

GPUInfo::~GPUInfo() = default;

GPUInfo::GPUDevice& GPUInfo::active_gpu() {
  return const_cast<GPUInfo::GPUDevice&>(
      const_cast<const GPUInfo&>(*this).active_gpu());
}

const GPUInfo::GPUDevice& GPUInfo::active_gpu() const {
  if (gpu.active || secondary_gpus.empty()) return gpu;
  for (const auto& secondary_gpu : secondary_gpus) {
    if (secondary_gpu.active) return secondary_gpu;
  }
  DVLOG(2) << "No active GPU found, returning primary GPU.";
  return gpu;
}

bool GPUInfo::IsInitialized() const { return gpu.vendor_id != 0; }

unsigned int GPUInfo::GpuCount() const {
  unsigned int gpu_count = 0;
  if (!gpu.IsSoftwareRenderer()) ++gpu_count;
  for (const auto& secondary_gpu : secondary_gpus) {
    if (!secondary_gpu.IsSoftwareRenderer()) ++gpu_count;
  }
  return gpu_count;
}

#if BUILDFLAG(IS_WIN)
GPUInfo::GPUDevice* GPUInfo::FindGpuByLuid(DWORD low_part, LONG high_part) {
  if (gpu.luid.LowPart == low_part && gpu.luid.HighPart == high_part)
    return &gpu;
  for (auto& device : secondary_gpus) {
    if (device.luid.LowPart == low_part && device.luid.HighPart == high_part)
      return &device;
  }
  return nullptr;
}
#endif  // BUILDFLAG(IS_WIN)

void GPUInfo::EnumerateFields(Enumerator* enumerator) const {
  struct GPUInfoKnownFields {
    base::TimeDelta initialization_time;
    bool optimus;
    bool amd_switchable;
    GPUDevice gpu;
    std::vector<GPUDevice> secondary_gpus;
    std::string machine_model_name;
    std::string machine_model_version;
    uint32_t target_cpu_bits;
  };

  // If this assert fails then most likely something below needs to be updated.
  // Note that this assert is only approximate. If a new field is added to
  // GPUInfo which fits within the current padding then it will not be caught.
  static_assert(
      sizeof(GPUInfo) == sizeof(GPUInfoKnownFields),
      "fields have changed in GPUInfo, GPUInfoKnownFields must be updated");

  // Required fields (according to DevTools protocol) first.
  enumerator->AddString("machineModelName", machine_model_name);
  enumerator->AddString("machineModelVersion", machine_model_version);
  EnumerateGPUDevice(gpu, enumerator);
  for (const auto& secondary_gpu : secondary_gpus)
    EnumerateGPUDevice(secondary_gpu, enumerator);

  enumerator->BeginAuxAttributes();
  enumerator->AddTimeDeltaInSecondsF("initializationTime", initialization_time);
  enumerator->AddBool("optimus", optimus);
  enumerator->AddBool("amdSwitchable", amd_switchable);
  enumerator->AddInt("targetCpuBits", static_cast<int>(target_cpu_bits));
  enumerator->EndAuxAttributes();
}

}  // namespace tachyon::device::gpu
