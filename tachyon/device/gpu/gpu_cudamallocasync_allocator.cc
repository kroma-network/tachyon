/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tachyon/device/gpu/gpu_cudamallocasync_allocator.h"

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#ifdef TACHYON_CUDA
// #include "tensorflow/compiler/xla/stream_executor/cuda/cuda_activation.h"
#include "third_party/gpus/cuda/include/cuda.h"
#endif  // TACHYON_CUDA

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
// #include "tensorflow/compiler/xla/stream_executor/device_id_utils.h"
// #include "tensorflow/compiler/xla/stream_executor/gpu/gpu_init.h"
// #include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tachyon/device/gpu/gpu_logging.h"

namespace tachyon::device::gpu {

/*
TODO(chokobole):
#if TACHYON_CUDA
static std::string GetCudaErrorMessage(CUresult result) {
  const char* error;
  cuGetErrorString(result, &error);
  const char* name;
  cuGetErrorName(result, &name);
  return absl::StrCat("CUDA error: ", error ? error : "<unknown>", " (",
                      name ? name : "Unknown", ")");
}
#endif  // TACHYON_CUDA
*/

void GpuCudaMallocAsyncAllocator::PrintAllocatorStatisticsNoLock() {
  std::map<size_t, int> size_map_histogram;
  std::vector<std::string> ptr_size_string;
  for (auto p : size_map_) {
    if (VLOG_IS_ON(8)) {
      ptr_size_string.push_back(
          absl::StrCat("(", absl::Hex(p.first), ",", p.second) + ")");
    }
    size_map_histogram[p.second]++;
  }
  LOG(ERROR) << "Histogram of current allocation: (allocation_size_in_bytes, "
             << "nb_allocation_of_that_sizes), ...;";
  for (auto p : size_map_histogram) {
    LOG(ERROR) << p.first << ", " << p.second;
  }

  VLOG(8) << "\nThe sorted list of (ptr,size):";
  VLOG(8) << absl::StrJoin(ptr_size_string, ",");

#if TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED
#if CUDA_VERSION >= 11030
  cuuint64_t mem_reserved_current;
  if (auto result = cuMemPoolGetAttribute(
          pool_, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT, &mem_reserved_current)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << GetCudaErrorMessage(result);
  }
  cuuint64_t mem_used_current;
  if (auto result = cuMemPoolGetAttribute(
          pool_, CU_MEMPOOL_ATTR_USED_MEM_CURRENT, &mem_used_current)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << GetCudaErrorMessage(result);
  }
  cuuint64_t mem_reserved_high;
  if (auto result = cuMemPoolGetAttribute(
          pool_, CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH, &mem_reserved_high)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << GetCudaErrorMessage(result);
  }
  cuuint64_t mem_used_high;
  if (auto result = cuMemPoolGetAttribute(pool_, CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                                          &mem_used_high)) {
    LOG(ERROR) << "Error while fetching extra cudaMallocAsync pool attribute: "
               << GetCudaErrorMessage(result);
  }
  LOG(ERROR) << "CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT: "
             << mem_reserved_current;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_USED_MEM_CURRENT: " << mem_used_current;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH: " << mem_reserved_high;
  LOG(ERROR) << "CU_MEMPOOL_ATTR_USED_MEM_HIGH: " << mem_used_high;
#endif
#endif  // TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED
}

void GpuCudaMallocAsyncAllocator::PrintAllocatorStatistics() {
  absl::MutexLock lock(&lock_);
  PrintAllocatorStatisticsNoLock();
}

std::atomic<int> GpuCudaMallocAsyncAllocator::number_instantiated_(0);

GpuCudaMallocAsyncAllocator::GpuCudaMallocAsyncAllocator(
    PlatformDeviceId platform_device_id, size_t pool_size, bool reserve_memory,
    bool compute_stats)
    : name_(absl::StrCat("gpu_async_", platform_device_id)),
      reserve_memory_(reserve_memory) {
  ++number_instantiated_;

  // Stop clang from complaining about unused private fields when
  // TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED is not defined.
  (void)reserve_memory_;

#if TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED
  stream_exec_ = DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(),
                                                           platform_device_id)
                     .value();
  // Initialized here as it only exist if compiled with a recent
  // enough CUDA.
  pool_ = nullptr;
  cuda_stream_ = nullptr;
  int driverVersion;
  cuDriverGetVersion(&driverVersion);
  VLOG(2) << "DRIVER VERSION: " << driverVersion;
  if (driverVersion < 11020) {
    LOG(FATAL)  // Crash OK.
        << "Disable cuda_malloc_async or update your CUDA driver to a version"
        << " compatible with CUDA 11.2 or higher."
        << " We detected a version compatible with: " << driverVersion;
  }

  // WAR an CUDA 11.2 driver bug for multiple-GPU. It currently
  // request that the context on GPU 0 is initialized. Which isn't the
  // case for Tachyon+horovod.
  if (platform_device_id.value() > 0 && driverVersion < 11030) {
    CUcontext pctx;  // We loose track of it. But this is fine.
    if (auto result = cuDevicePrimaryCtxRetain(&pctx, 0))
      LOG(FATAL)  // Crash OK.
          << "Failed to retain context: " << GetCudaErrorMessage(result);
  }

  cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};

  // Check the CUDA runtime is recent enough.
  if (auto status2 = cuDriverGetVersion(&driverVersion)) {
    LOG(FATAL)  // Crash OK.
        << "Error while fetching driver version: "
        << GetCudaErrorMessage(status2);
  }

  // Check that cudaMallocAsync is supported.
  int cuda_malloc_async_supported;
  if (auto status =
          cuDeviceGetAttribute(&cuda_malloc_async_supported,
                               CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                               platform_device_id.value())) {
    LOG(FATAL)  // Crash OK.
        << "On device: " << platform_device_id.value()
        << " Current driver: " << driverVersion
        << ". Failed to get device attribute : " << GetCudaErrorMessage(status);
  }
  if (!cuda_malloc_async_supported)
    LOG(FATAL)  // Crash OK.
        << "TF_GPU_ALLOCATOR=cuda_malloc_async isn't currently supported on "
        << "GPU id " << platform_device_id.value() << ":"
        << " Possible causes: device not supported (request SM60+), driver too "
           "old, "
        << " OS not supported, CUDA version too old(request CUDA11.2+).";

  if (auto status =
          cuDeviceGetDefaultMemPool(&pool_, platform_device_id.value()))
    LOG(FATAL) <<  // Crash OK.
        "Failed to get default CUDA pool: " << GetCudaErrorMessage(status);

  VLOG(1) << Name() << " CudaMallocAsync initialized on platform: "
          << platform_device_id.value() << " with pool size of: " << pool_size
          << " this ptr: " << this;
  uint64_t pool_size_64 = pool_size;
  if (auto status = cuMemPoolSetAttribute(
          pool_, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &pool_size_64))
    LOG(FATAL) <<  // Crash OK.
        "Failed to set CUDA pool attribute: " << GetCudaErrorMessage(status);

  if (compute_stats) {
    stats_ = std::make_unique<AllocatorStats>();
    stats_->bytes_limit = static_cast<int64_t>(pool_size);
  }  // If not set, it means we do not compute stats.

  // If in TF_DETERMINISTIC_ALLOCATOR is set, then make the allocator behave
  // determistically.
  bool deterministic = false;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_DETERMINISTIC_ALLOCATOR",
                                      /*default_val=*/false, &deterministic));
  if (deterministic) {
    int disable = 0;
    if (auto status = cuMemPoolSetAttribute(
            pool_, CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, &disable)) {
      LOG(FATAL) <<  // Crash OK.
          "Failed to set CUDA pool attribute: " << GetCudaErrorMessage(status);
    }
    if (auto status = cuMemPoolSetAttribute(
            pool_, CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
            &disable)) {
      LOG(FATAL) <<  // Crash OK.
          "Failed to set CUDA pool attribute: " << GetCudaErrorMessage(status);
    }
  }

  // Set read/write access to all GPUs.
  static auto* all_pools_ = new std::vector<CUmemoryPool*>();
  static auto* all_ids_ = new std::vector<tsl::PlatformDeviceId>();
  DCHECK(all_pools_->size() == all_ids_->size());
  for (int i = 0; i < all_pools_->size(); ++i) {
    // Set the current pool access to the previous GPUs.
    CUmemAccessDesc map;
    map.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    map.location.id = (*all_ids_)[i].value();

    map.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    VLOG(2) << "Setting access of the current pool to "
            << " location id: " << map.location.id;
    int canAccessPeer;
    if (auto status = cuDeviceCanAccessPeer(
            &canAccessPeer, platform_device_id.value(), map.location.id)) {
      pool_ = nullptr;
      LOG(FATAL)  // Crash OK.
          << "cuDeviceCanAccessPeer failed to know if GPU id "
          << map.location.id << " can access GPU id "
          << platform_device_id.value() << ": " << GetCudaErrorMessage(status);
    }
    if (canAccessPeer == 1) {
      if (auto status = cuMemPoolSetAccess(pool_, &map, 1)) {
        pool_ = nullptr;
        LOG(FATAL)  // Crash OK.
            << "Error when setting access to the pool id: " << i
            << " location id: " << map.location.id
            << " error: " << GetCudaErrorMessage(status);
      }
    }

    // Set the previous pools access to the current GPU.
    map.location.id = platform_device_id.value();

    VLOG(2) << "Set access to the pool id: " << i
            << " location id: " << map.location.id;
    if (auto status = cuDeviceCanAccessPeer(&canAccessPeer, i,
                                            platform_device_id.value())) {
      pool_ = nullptr;
      LOG(FATAL)  // Crash OK.
          << "cuDeviceCanAccessPeer failed: " << GetCudaErrorMessage(status);
    }
    if (canAccessPeer == 1) {
      if (auto status = cuMemPoolSetAccess(*(*all_pools_)[i], &map, 1)) {
        pool_ = nullptr;
        LOG(FATAL)  // Crash OK.
            << "Error when setting access to the pool id: " << i
            << " location id: " << map.location.id
            << " error: " << GetCudaErrorMessage(status);
      }
    }
  }
  all_pools_->push_back(&pool_);
  all_ids_->push_back(platform_device_id);

  VLOG(2) << Name() << " GpuCudaMallocAsyncAllocator PoolSize " << pool_size;
#else   // TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED
  LOG(FATAL) << "GpuCudaMallocAsyncAllocator requires CUDA 11.2+";  // Crash OK.
#endif  // TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED
}

GpuCudaMallocAsyncAllocator::~GpuCudaMallocAsyncAllocator() {}

void* GpuCudaMallocAsyncAllocator::AllocateRaw(size_t alignment,
                                               size_t num_bytes) {
#if TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED
  CHECK(cuda_stream_ != nullptr)
      << "A stream must be added to the GpuCudaMallocAsync allocator";
  if (pool_ == nullptr) {
    LOG(FATAL)  // Crash OK.
        << "The instantiation of GpuCudaMallocAsyncAllocator failed."
        << " See previous errors.";
  }
  // The lock is only needed when stats are enabled, but it must be around
  // the cuMemAllocFromPoolAsync call as well to ensure consistency of the stats
  // update.
  std::unique_lock<absl::Mutex> lock(lock_, std::defer_lock);
  if (stats_) {
    lock.lock();
  }
  cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  void* ptr = nullptr;
  if (auto result =
          cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&ptr),
                                  num_bytes, pool_, cuda_stream_)) {
    size_t free, total;
    cuMemGetInfo(&free, &total);
    LOG(ERROR) << Name() << " cuMemAllocAsync failed to allocate " << num_bytes
               << " bytes: " << GetCudaErrorMessage(result)
               << "\n Reported by CUDA: Free memory/Total memory: " << free
               << "/" << total;
    if (stats_) {
      LOG(ERROR) << "Stats: " << stats_->DebugString();
      PrintAllocatorStatisticsNoLock();
    }

    return nullptr;
  }

  // Update stats.
  if (stats_) {
    ++(stats_->num_allocs);
    stats_->bytes_in_use += num_bytes;
    if (stats_->bytes_in_use > stats_->peak_bytes_in_use) {
      VLOG(9) << "New Peak memory usage of " << stats_->bytes_in_use
              << " bytes.";
    }
    stats_->peak_bytes_in_use =
        std::max(stats_->peak_bytes_in_use, stats_->bytes_in_use);
    stats_->largest_alloc_size =
        std::max<std::size_t>(stats_->largest_alloc_size, num_bytes);
    bool ptr_inserted = size_map_.emplace(ptr, num_bytes).second;
    DCHECK(ptr_inserted);
  }
  VLOG(10) << Name() << " Allocated " << num_bytes << " at " << ptr;
  return ptr;
#else   // TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED
  return nullptr;
#endif  // TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED
}
void GpuCudaMallocAsyncAllocator::DeallocateRaw(void* ptr) {
#if TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED
  if (ptr == nullptr) return;
  // The lock is only needed when stats are enabled, but it must be around
  // the cuMemFreeAsync call as well to ensure consistency of the stats update.
  std::unique_lock<absl::Mutex> lock(lock_, std::defer_lock);
  if (stats_) {
    lock.lock();
  }
  if (auto result = cuMemFreeAsync(reinterpret_cast<const CUdeviceptr&>(ptr),
                                   cuda_stream_)) {
    if (result == CUDA_ERROR_DEINITIALIZED) {
      // It happens with multi-GPU that Tachyon free the GPU allocation after
      // the driver is unloaded. It is safe to ignore this error here.
      // TODO: Find how to fix the shutdown steps in Tachyon.
      VLOG(1) << "Ignoring CUDA error: " << GetCudaErrorMessage(result);
    } else {
      size_t free, total;
      cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
      cuMemGetInfo(&free, &total);
      LOG(ERROR) << "cudaFreeAsync failed to free " << ptr << ": "
                 << GetCudaErrorMessage(result)
                 << "\n Free memory/Total memory: " << free << "/" << total;
      if (stats_) {
        LOG(ERROR) << "Stats: " << stats_->DebugString();
      }
    }
  }

  // Updates the stats.
  if (stats_) {
    DCHECK(size_map_.contains(ptr));
    size_t size = size_map_[ptr];
    stats_->bytes_in_use -= size;
    size_map_.erase(ptr);
  }

  VLOG(10) << Name() << " Freed ptr: " << ptr;
#endif  // TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED
}

bool GpuCudaMallocAsyncAllocator::TracksAllocationSizes() const {
  return static_cast<bool>(stats_);
}

size_t GpuCudaMallocAsyncAllocator::RequestedSize(const void* ptr) const {
  if (!stats_ || !ptr) return 0;
  absl::MutexLock l(&lock_);
  return size_map_.at(ptr);
}

size_t GpuCudaMallocAsyncAllocator::AllocatedSize(const void* ptr) const {
  if (!stats_ || !ptr) return 0;
  absl::MutexLock l(&lock_);
  return size_map_.at(ptr);
}

std::optional<AllocatorStats> GpuCudaMallocAsyncAllocator::GetStats() {
  if (!stats_) return std::nullopt;
  absl::MutexLock l(&lock_);
  return *stats_;
}

bool GpuCudaMallocAsyncAllocator::ClearStats() {
  if (!stats_) return false;
  absl::MutexLock l(&lock_);
  stats_->num_allocs = 0;
  stats_->peak_bytes_in_use = stats_->bytes_in_use;
  stats_->largest_alloc_size = 0;
  return true;
}

void GpuCudaMallocAsyncAllocator::SetStreamAndPreallocateMemory(void* stream) {
#if TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED
  CUstream new_cuda_stream = *(static_cast<CUstream*>(stream));
  // We don't need to re-set the CUDA stream if this is the same stream
  if (cuda_stream_ != nullptr && new_cuda_stream != cuda_stream_) {
    LOG(FATAL) <<  // Crash OK.
        "Trying to set the stream twice. This isn't supported. ";
  }

  uint64_t pool_size_64 = 0;
  if (auto status = cuMemPoolGetAttribute(
          pool_, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &pool_size_64)) {
    LOG(FATAL) <<  // Crash OK.
        "Failed to get CUDA pool attribute: " << GetCudaErrorMessage(status);
  }
  cuda_stream_ = new_cuda_stream;
  int64_t prealloc_size = 0;
  // TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=-1 is a special value that
  // preallocates the total pool size.
  TF_CHECK_OK(tsl::ReadInt64FromEnvVar(
      "TACHYON_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC", 0, &prealloc_size));
  if (prealloc_size == -1) {
    prealloc_size = pool_size_64;
  } else if (reserve_memory_) {
    prealloc_size = pool_size_64;
  }

  if (prealloc_size != 0) {
    void* ptr = AllocateRaw(0, prealloc_size);
    DeallocateRaw(ptr);
    VLOG(2) << Name() << " GpuCudaMallocAsyncAllocator reserved the pool for "
            << prealloc_size << " bytes"
            << ". First ptr: " << ptr;
    ClearStats();
  }
#endif
}

}  // namespace tachyon::device::gpu
