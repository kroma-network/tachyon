/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tachyon/device/gpu/cuda/cuda_driver.h"

#define FAIL_IF_CUDA_RES_ERROR(expr, ...)                   \
  do {                                                      \
    CUresult _res = (expr);                                 \
    if (ABSL_PREDICT_FALSE(_res != CUDA_SUCCESS)) {         \
      LOG(FATAL) << absl::StrCat(__VA_ARGS__) << ": "       \
                 << ::tachyon::device::gpu::ToString(_res); \
    }                                                       \
  } while (0)

namespace tachyon {
namespace device {
namespace gpu {

// static
absl::Mutex CreatedContexts::mu_{absl::kConstInit};

// static
int64_t CreatedContexts::next_id_ = 1;  // 0 means "no context"

ScopedActivateContext::ScopedActivateContext(GpuContext* cuda_context) {
  NOTIMPLEMENTED();
}

ScopedActivateContext::~ScopedActivateContext() { NOTIMPLEMENTED(); }

}  // namespace gpu

namespace cuda {

CUcontext CurrentContextOrDie() {
  CUcontext current = nullptr;
  FAIL_IF_CUDA_RES_ERROR(cuCtxGetCurrent(&current),
                         "Failed to query current context");
  return current;
}

}  // namespace cuda
}  // namespace device
}  // namespace tachyon
