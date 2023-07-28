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

// CUDA userspace driver library wrapper functionality.

#ifndef TACHYON_DEVICE_GPU_CUDA_CUDA_DRIVER_H_
#define TACHYON_DEVICE_GPU_CUDA_CUDA_DRIVER_H_

#include "absl/container/node_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuda.h"

#include "tachyon/base/logging.h"
#include "tachyon/device/gpu/gpu_driver.h"
#include "tachyon/export.h"

namespace tachyon::device {
namespace gpu {

// Formats CUresult to output prettified values into a log stream.
static std::string ToString(CUresult result) {
  const char* error_name;
  if (cuGetErrorName(result, &error_name)) {
    return absl::StrCat("UNKNOWN ERROR (", static_cast<int>(result), ")");
  }
  const char* error_string;
  if (cuGetErrorString(result, &error_string)) {
    return error_name;
  }
  return absl::StrCat(error_name, ": ", error_string);
}

// CUDAContext wraps a cuda CUcontext handle, and includes a unique id. The
// unique id is positive, and ids are not repeated within the process.
class GpuContext {
 public:
  GpuContext(CUcontext context, int64_t id) : context_(context), id_(id) {}

  CUcontext context() const { return context_; }
  int64_t id() const { return id_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  CUcontext const context_;
  const int64_t id_;
};

// Manages the singleton map of contexts that we've created, mapping
// from the CUcontext to the GpuContext* that we pass around internally.
// This also manages assignment of unique ids to GpuContexts, to allow
// for fast comparison of a context against the current context.
//
// CUDA-runtime-created contexts are avoided, if triple angle
// brace launches are required, by using the scoped activations in
// gpu/gpu_activation.h.
class CreatedContexts {
 public:
  // Returns whether context is a member of the live set.
  static bool Has(CUcontext context) {
    absl::ReaderMutexLock lock(&mu_);
    return Live()->find(context) != Live()->end();
  }

  // Adds context to the live set, or returns it if it's already present.
  static GpuContext* Add(CUcontext context, int device_ordinal) {
    CHECK(context != nullptr);
    absl::MutexLock lock(&mu_);

    auto insert_result = Live()->insert(std::make_pair(context, nullptr));
    auto it = insert_result.first;
    if (insert_result.second) {
      // context was not present in the map.  Add it.
      it->second = std::make_unique<GpuContext>(context, next_id_++);
      (*LiveOrdinal())[device_ordinal].push_back(context);
    }
    return it->second.get();
  }

  // Removes context from the live set.
  static void Remove(CUcontext context) {
    CHECK(context != nullptr);
    absl::MutexLock lock(&mu_);
    auto it = Live()->find(context);
    CHECK(it != Live()->end()) << context;
    Live()->erase(it);
    for (auto p : (*LiveOrdinal())) {
      auto it2 = std::find(p.second.begin(), p.second.end(), context);
      if (it2 != p.second.end()) {
        p.second.erase(it2, it2++);
        if (p.second.empty()) {
          LiveOrdinal()->erase(p.first);
        }
        break;
      }
    }
  }

  // Return the context associated to that ptr.
  static CUcontext GetAnyContext(void* ptr) {
    absl::ReaderMutexLock lock(&mu_);
    int device_ordinal;
    CUresult result = cuPointerGetAttribute(static_cast<void*>(&device_ordinal),
                                            CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                            reinterpret_cast<CUdeviceptr>(ptr));
    if (result != CUDA_SUCCESS) {
      LOG(FATAL) << "Not able to get the device_ordinal for ptr: " << ptr
                 << ". Error: " << ToString(result);
    }
    CHECK_EQ(LiveOrdinal()->count(device_ordinal), static_cast<size_t>(1));
    CHECK(!LiveOrdinal()->at(device_ordinal).empty())
        << "Need at least one context.";
    return LiveOrdinal()->at(device_ordinal)[0];
  }

 private:
  // Returns the live map singleton.
  static absl::node_hash_map<CUcontext, std::unique_ptr<GpuContext>>* Live() {
    static auto singleton =
        new absl::node_hash_map<CUcontext, std::unique_ptr<GpuContext>>;
    return singleton;
  }
  static absl::node_hash_map<int, std::vector<CUcontext>>* LiveOrdinal() {
    static auto singleton =
        new absl::node_hash_map<int, std::vector<CUcontext>>;
    return singleton;
  }

  // Lock that guards access-to/mutation-of the live set.
  static absl::Mutex mu_;
  static int64_t next_id_;
};

}  // namespace gpu

namespace cuda {

using MemorySpace = gpu::MemorySpace;

using CUDADriver = gpu::GpuDriver;

using ScopedActivateContext = gpu::ScopedActivateContext;

using CudaContext = gpu::GpuContext;

// Returns the current context set in CUDA. This is done by calling the cuda
// driver (e.g., this value is not our cached view of the current context).
CUcontext CurrentContextOrDie();

}  // namespace cuda
}  // namespace tachyon::device

#endif  // TACHYON_DEVICE_GPU_CUDA_CUDA_DRIVER_H_
