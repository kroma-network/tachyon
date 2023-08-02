/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Classes to maintain a static registry of memory allocator factories.
#ifndef TACHYON_DEVICE_ALLOCATOR_REGISTRY_H_
#define TACHYON_DEVICE_ALLOCATOR_REGISTRY_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

#include "tachyon/device/allocator.h"
#include "tachyon/export.h"

namespace tachyon::device {

class TACHYON_EXPORT AllocatorFactory {
 public:
  virtual ~AllocatorFactory() {}

  // Returns true if the factory will create a functionally different
  // SubAllocator for different (legal) values of numa_node.
  virtual bool NumaEnabled() { return false; }

  // Create an Allocator.
  virtual Allocator* CreateAllocator() = 0;

  // Create a SubAllocator. If NumaEnabled() is true, then returned SubAllocator
  // will allocate memory local to numa_node.  If numa_node == kNUMANoAffinity
  // then allocated memory is not specific to any NUMA node.
  virtual SubAllocator* CreateSubAllocator(int numa_node) = 0;
};

// ProcessState is defined in a package that cannot be a dependency of
// framework.  This definition allows us to access the one method we need.
class TACHYON_EXPORT ProcessStateInterface {
 public:
  virtual ~ProcessStateInterface() {}
  virtual Allocator* GetCPUAllocator(int numa_node) = 0;
};

// A singleton registry of AllocatorFactories.
//
// Allocators should be obtained through ProcessState or cpu_allocator()
// (deprecated), not directly through this interface.  The purpose of this
// registry is to allow link-time discovery of multiple AllocatorFactories among
// which ProcessState will obtain the best fit at startup.
class TACHYON_EXPORT AllocatorFactoryRegistry {
 public:
  AllocatorFactoryRegistry() {}
  AllocatorFactoryRegistry(const AllocatorFactoryRegistry& other) = delete;
  AllocatorFactoryRegistry& operator=(const AllocatorFactoryRegistry& other) =
      delete;
  ~AllocatorFactoryRegistry() {}

  void Register(const char* source_file, int source_line,
                const std::string& name, int priority,
                AllocatorFactory* factory);

  // Returns 'best fit' Allocator.  Find the factory with the highest priority
  // and return an allocator constructed by it.  If multiple factories have
  // been registered with the same priority, picks one by unspecified criteria.
  Allocator* GetAllocator();

  // Returns 'best fit' SubAllocator.  First look for the highest priority
  // factory that is NUMA-enabled.  If none is registered, fall back to the
  // highest priority non-NUMA-enabled factory.  If NUMA-enabled, return a
  // SubAllocator specific to numa_node, otherwise return a NUMA-insensitive
  // SubAllocator.
  SubAllocator* GetSubAllocator(int numa_node);

  // Returns the singleton value.
  static AllocatorFactoryRegistry& Get();

  ProcessStateInterface* process_state() const { return process_state_; }

 protected:
  ProcessStateInterface* process_state_ = nullptr;

 private:
  absl::Mutex mu_;
  bool first_alloc_made_ = false;
  struct FactoryEntry {
    const char* source_file;
    int source_line;
    std::string name;
    int priority;
    std::unique_ptr<AllocatorFactory> factory;
    std::unique_ptr<Allocator> allocator;
    // Index 0 corresponds to kNUMANoAffinity, other indices are (numa_node +
    // 1).
    std::vector<std::unique_ptr<SubAllocator>> sub_allocators;
  };
  std::vector<FactoryEntry> factories_ ABSL_GUARDED_BY(mu_);

  // Returns any FactoryEntry registered under 'name' and 'priority',
  // or 'nullptr' if none found.
  const FactoryEntry* FindEntry(const std::string& name, int priority) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
};

class TACHYON_EXPORT AllocatorFactoryRegistration {
 public:
  AllocatorFactoryRegistration(const char* file, int line,
                               const std::string& name, int priority,
                               AllocatorFactory* factory) {
    AllocatorFactoryRegistry::Get().Register(file, line, name, priority,
                                             factory);
  }
};

#define REGISTER_MEM_ALLOCATOR(name, priority, factory)                     \
  REGISTER_MEM_ALLOCATOR_UNIQ_HELPER(__COUNTER__, __FILE__, __LINE__, name, \
                                     priority, factory)

#define REGISTER_MEM_ALLOCATOR_UNIQ_HELPER(ctr, file, line, name, priority, \
                                           factory)                         \
  REGISTER_MEM_ALLOCATOR_UNIQ(ctr, file, line, name, priority, factory)

#define REGISTER_MEM_ALLOCATOR_UNIQ(ctr, file, line, name, priority, factory) \
  static AllocatorFactoryRegistration allocator_factory_reg_##ctr(            \
      file, line, name, priority, new factory)

}  // namespace tachyon::device

#endif  // TACHYON_DEVICE_ALLOCATOR_REGISTRY_H_
