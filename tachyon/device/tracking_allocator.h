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

#ifndef TACHYON_DEVICE_TRACKING_ALLOCATOR_H_
#define TACHYON_DEVICE_TRACKING_ALLOCATOR_H_

#include <optional>
#include <unordered_map>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"

#include "tachyon/base/time/time.h"
#include "tachyon/device/allocator.h"

namespace tachyon::device {

// TrackingAllocator is a wrapper for an Allocator. It keeps a running
// count of the number of bytes allocated through the wrapper. It is
// used by the Executor to "charge" allocations to particular Op
// executions. Each Op gets a separate TrackingAllocator wrapper
// around the underlying allocator.
//
// The implementation assumes the invariant that all calls to
// AllocateRaw by an Op (or work items spawned by the Op) will occur
// before the Op's Compute method returns. Thus the high watermark is
// established once Compute returns.
//
// DeallocateRaw can be called long after the Op has finished,
// e.g. when an output tensor is deallocated, and the wrapper cannot
// be deleted until the last of these calls has occurred.  The
// TrackingAllocator keeps track of outstanding calls using a
// reference count, and deletes itself once the last call has been
// received and the high watermark has been retrieved.
struct TACHYON_EXPORT AllocRecord {
  AllocRecord(int64_t alloc_bytes, base::TimeTicks alloc_time)
      : alloc_bytes(alloc_bytes), alloc_time(alloc_time) {}
  AllocRecord() : AllocRecord(0, base::TimeTicks()) {}

  int64_t alloc_bytes;
  base::TimeTicks alloc_time;
};

class TACHYON_EXPORT TrackingAllocator : public Allocator {
 public:
  explicit TrackingAllocator(Allocator* allocator, bool track_ids);
  std::string Name() override { return allocator_->Name(); }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return AllocateRaw(alignment, num_bytes, AllocationAttributes());
  }
  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;
  void DeallocateRaw(void* ptr) override;
  bool TracksAllocationSizes() const override;
  size_t RequestedSize(const void* ptr) const override;
  size_t AllocatedSize(const void* ptr) const override;
  int64_t AllocationId(const void* ptr) const override;
  std::optional<AllocatorStats> GetStats() override;
  bool ClearStats() override;

  AllocatorMemoryType GetMemoryType() const override {
    return allocator_->GetMemoryType();
  }

  // If the underlying allocator tracks allocation sizes, this returns
  // a tuple where the first value is the total number of bytes
  // allocated through this wrapper, the second value is the high
  // watermark of bytes allocated through this wrapper and the third value is
  // the allocated bytes through this wrapper that are still alive. If the
  // underlying allocator does not track allocation sizes the first
  // value is the total number of bytes requested through this wrapper
  // and the second and the third are 0.
  //
  std::tuple<size_t, size_t, size_t> GetSizes();
  // After GetRecordsAndUnRef is called, the only further calls allowed
  // on this wrapper are calls to DeallocateRaw with pointers that
  // were allocated by this wrapper and have not yet been
  // deallocated. After this call completes and all allocated pointers
  // have been deallocated the wrapper will delete itself.
  absl::InlinedVector<AllocRecord, 4> GetRecordsAndUnRef();
  // Returns a copy of allocation records collected so far.
  absl::InlinedVector<AllocRecord, 4> GetCurrentRecords();

 protected:
  ~TrackingAllocator() override {}

 private:
  bool UnRef() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  Allocator* allocator_;  // not owned.
  mutable absl::Mutex mu_;
  // the number of calls to AllocateRaw that have not yet been matched
  // by a corresponding call to DeAllocateRaw, plus 1 if the Executor
  // has not yet read out the high watermark.
  int ref_ ABSL_GUARDED_BY(mu_);
  // the current number of outstanding bytes that have been allocated
  // by this wrapper, or 0 if the underlying allocator does not track
  // allocation sizes.
  size_t allocated_ ABSL_GUARDED_BY(mu_);
  // the maximum number of outstanding bytes that have been allocated
  // by this wrapper, or 0 if the underlying allocator does not track
  // allocation sizes.
  size_t high_watermark_ ABSL_GUARDED_BY(mu_);
  // the total number of bytes that have been allocated by this
  // wrapper if the underlying allocator tracks allocation sizes,
  // otherwise the total number of bytes that have been requested by
  // this allocator.
  size_t total_bytes_ ABSL_GUARDED_BY(mu_);

  absl::InlinedVector<AllocRecord, 4> allocations_ ABSL_GUARDED_BY(mu_);

  // Track allocations locally if requested in the constructor and the
  // underlying allocator doesn't already do it for us.
  const bool track_sizes_locally_;
  struct Chunk {
    size_t requested_size;
    size_t allocated_size;
    int64_t allocation_id;
  };
  std::unordered_map<const void*, Chunk> in_use_ ABSL_GUARDED_BY(mu_);
  int64_t next_allocation_id_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tachyon::device

#endif  // TACHYON_DEVICE_TRACKING_ALLOCATOR_H_
