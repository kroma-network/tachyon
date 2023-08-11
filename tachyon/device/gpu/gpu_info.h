// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_DEVICE_GPU_GPU_INFO_H_
#define TACHYON_DEVICE_GPU_GPU_INFO_H_

// Provides access to the GPU information for the system
// on which chrome is currently running.

#include <stdint.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/time/time.h"
#include "tachyon/build/build_config.h"

namespace tachyon::device::gpu {

// These values are persistent to logs. Entries should not be renumbered and
// numeric values should never be reused.
// This should match enum IntelGpuSeriesType in
//  \tools\metrics\histograms\enums.xml
enum class IntelGpuSeriesType {
  kUnknown = 0,
  // Intel 4th gen
  kBroadwater = 16,
  kEaglelake = 17,
  // Intel 5th gen
  kIronlake = 18,
  // Intel 6th gen
  kSandybridge = 1,
  // Intel 7th gen
  kBaytrail = 2,
  kIvybridge = 3,
  kHaswell = 4,
  // Intel 8th gen
  kCherrytrail = 5,
  kBroadwell = 6,
  // Intel 9th gen
  kApollolake = 7,
  kSkylake = 8,
  kGeminilake = 9,
  kKabylake = 10,
  kAmberlake = 23,
  kCoffeelake = 11,
  kWhiskeylake = 12,
  kCometlake = 13,
  // Intel 10th gen
  kCannonlake = 14,
  // Intel 11th gen
  kIcelake = 15,
  kElkhartlake = 19,
  kJasperlake = 20,
  // Intel 12th gen
  kTigerlake = 21,
  kRocketlake = 24,
  kDG1 = 25,
  kAlderlake = 22,
  kAlchemist = 26,
  kRaptorlake = 27,
  kMeteorlake = 28,
  // Please also update |gpu_series_map| in process_json.py.
  kMaxValue = kMeteorlake,
};

struct TACHYON_EXPORT GPUInfo {
  struct TACHYON_EXPORT GPUDevice {
    GPUDevice();
    GPUDevice(const GPUDevice& other);
    GPUDevice(GPUDevice&& other) noexcept;
    ~GPUDevice() noexcept;
    GPUDevice& operator=(const GPUDevice& other);
    GPUDevice& operator=(GPUDevice&& other) noexcept;

    bool IsSoftwareRenderer() const;

    // The DWORD (uint32_t) representing the graphics card vendor id.
    uint32_t vendor_id = 0u;

    // The DWORD (uint32_t) representing the graphics card device id.
    // Device ids are unique to vendor, not to one another.
    uint32_t device_id = 0u;

#if BUILDFLAG(IS_WIN) || BUILDFLAG(IS_CHROMEOS)
    // The graphics card revision number.
    uint32_t revision = 0u;
#endif

#if BUILDFLAG(IS_WIN)
    // The graphics card subsystem id.
    // The lower 16 bits represents the subsystem vendor id.
    uint32_t sub_sys_id = 0u;

    // The graphics card LUID. This is a unique identifier for the graphics card
    // that is guaranteed to be unique until the computer is restarted. The LUID
    // is used over the vendor id and device id because the device id is only
    // unique relative its vendor, not to each other. If there are more than one
    // of the same exact graphics card, they all have the same vendor id and
    // device id but different LUIDs.
    CHROME_LUID luid;
#endif  // BUILDFLAG(IS_WIN)

    // The 64-bit ID used for GPU selection by ANGLE_platform_angle_device_id.
    // On Mac this matches the registry ID of an IOGraphicsAccelerator2 or
    // AGXAccelerator.
    // On Windows this matches the concatenated LUID.
    uint64_t system_device_id = 0ULL;

    // Whether this GPU is the currently used one.
    // Currently this field is only supported and meaningful on OS X and on
    // Windows using Angle with D3D11.
    bool active = false;

    // The strings that describe the GPU.
    // In Linux these strings are obtained through libpci.
    // In Win/MacOSX, these two strings are not filled at the moment.
    // In Android, these are respectively GL_VENDOR and GL_RENDERER.
    std::string vendor_string;
    std::string device_string;

    std::string driver_vendor;
    std::string driver_version;

    // NVIDIA CUDA compute capability, major version. 0 if undetermined. Can be
    // used to determine the hardware generation that the GPU belongs to.
    int cuda_compute_capability_major = 0;
  };

  GPUInfo();
  GPUInfo(const GPUInfo& other);
  ~GPUInfo();

  // The currently active gpu.
  GPUDevice& active_gpu();
  const GPUDevice& active_gpu() const;

  bool IsInitialized() const;

  unsigned int GpuCount() const;

#if BUILDFLAG(IS_WIN)
  GPUDevice* FindGpuByLuid(DWORD low_part, LONG high_part);
#endif  // BUILDFLAG(IS_WIN)

  // The amount of time taken to get from the process starting to the message
  // loop being pumped.
  base::TimeDelta initialization_time;

  // Computer has NVIDIA Optimus
  bool optimus;

  // Computer has AMD Dynamic Switchable Graphics
  bool amd_switchable;

  // Primary GPU, for example, the discrete GPU in a dual GPU machine.
  GPUDevice gpu;

  // Secondary GPUs, for example, the integrated GPU in a dual GPU machine.
  std::vector<GPUDevice> secondary_gpus;

  // The machine model identifier. They can contain any character, including
  // whitespaces.  Currently it is supported on MacOSX and Android.
  // Android examples: "Naxus 5", "XT1032".
  // On MacOSX, the version is stripped out of the model identifier, for
  // example, the original identifier is "MacBookPro7,2", and we put
  // "MacBookPro" as machine_model_name, and "7.2" as machine_model_version.
  std::string machine_model_name;

  // The version of the machine model. Currently it is supported on MacOSX.
  // See machine_model_name's comment.
  std::string machine_model_version;

  // Whether the gpu process is running in a sandbox.
  bool sandboxed;

#if defined(ARCH_CPU_64_BITS)
  uint32_t target_cpu_bits = 64;
#elif defined(ARCH_CPU_32_BITS)
  uint32_t target_cpu_bits = 32;
#elif defined(ARCH_CPU_31_BITS)
  uint32_t target_cpu_bits = 31;
#endif

  // Note: when adding new members, please remember to update EnumerateFields
  // in gpu_info.cc.

  // In conjunction with EnumerateFields, this allows the embedder to
  // enumerate the values in this structure without having to embed
  // references to its specific member variables. This simplifies the
  // addition of new fields to this type.
  class Enumerator {
   public:
    // The following methods apply to the "current" object. Initially this
    // is the root object, but calls to BeginGPUDevice/EndGPUDevice and
    // BeginAuxAttributes/EndAuxAttributes change the object to which these
    // calls should apply.
    virtual void AddInt64(const char* name, int64_t value) = 0;
    virtual void AddInt(const char* name, int value) = 0;
    virtual void AddString(const char* name, const std::string& value) = 0;
    virtual void AddBool(const char* name, bool value) = 0;
    virtual void AddTimeDeltaInSecondsF(const char* name,
                                        const base::TimeDelta& value) = 0;
    virtual void AddBinary(const char* name,
                           const absl::Span<const uint8_t>& blob) = 0;

    // Markers indicating that a GPUDevice is being described.
    virtual void BeginGPUDevice() = 0;
    virtual void EndGPUDevice() = 0;

    // Markers indicating that "auxiliary" attributes of the GPUInfo
    // (according to the DevTools protocol) are being described.
    virtual void BeginAuxAttributes() = 0;
    virtual void EndAuxAttributes() = 0;

   protected:
    virtual ~Enumerator() = default;
  };

  // Outputs the fields in this structure to the provided enumerator.
  void EnumerateFields(Enumerator* enumerator) const;
};

}  // namespace tachyon::device::gpu

#endif  // TACHYON_DEVICE_GPU_GPU_INFO_H_
