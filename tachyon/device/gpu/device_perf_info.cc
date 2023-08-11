// Copyright 2020 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/device/gpu/device_perf_info.h"

#include "absl/synchronization/mutex.h"

#include "tachyon/base/no_destructor.h"

namespace tachyon::device::gpu {

namespace {

std::optional<DevicePerfInfo> g_device_perf_info;

absl::Mutex* GetLock() {
  static base::NoDestructor<absl::Mutex> lock;
  return lock.get();
}

}  // namespace

std::optional<DevicePerfInfo> GetDevicePerfInfo() {
  absl::MutexLock lock(GetLock());
  return g_device_perf_info;
}

void SetDevicePerfInfo(const DevicePerfInfo& device_perf_info) {
  absl::MutexLock lock(GetLock());
  g_device_perf_info = device_perf_info;
}

}  // namespace tachyon::device::gpu
