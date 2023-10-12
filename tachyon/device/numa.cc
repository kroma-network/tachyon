#include "tachyon/device/numa.h"

#include <algorithm>

#include "tachyon/base/logging.h"
#include "tachyon/base/memory/aligned_memory.h"
#include "tachyon/build/build_config.h"

#if defined(TACHYON_USE_NUMA)
// NOLINTNEXTLINE(build/include_subdir)
#include "hwloc.h"  // from @hwloc
#endif

namespace tachyon::device {

#if BUILDFLAG(IS_WIN)

bool NUMAEnabled() {
  NOTIMPLEMENTED();
  return false;
}

int NUMANumNodes() { return 1; }

void NUMASetThreadNodeAffinity(int node) {}

int NUMAGetThreadNodeAffinity() { return kNUMANoAffinity; }

void* NUMAMalloc(int node, size_t size, int minimum_alignment) {
  return tsl::port::AlignedMalloc(size, minimum_alignment);
}

void NUMAFree(void* ptr, size_t size) { tsl::port::Free(ptr); }

int NUMAGetMemAffinity(const void* addr) { return kNUMANoAffinity; }

#else

#if defined(TACHYON_USE_NUMA)
namespace {
static hwloc_topology_t hwloc_topology_handle;

bool HaveHWLocTopology() {
  // One time initialization
  static bool init = []() {
    if (hwloc_topology_init(&hwloc_topology_handle)) {
      LOG(ERROR) << "Call to hwloc_topology_init() failed";
      return false;
    }
    if (hwloc_topology_load(hwloc_topology_handle)) {
      LOG(ERROR) << "Call to hwloc_topology_load() failed";
      return false;
    }
    return true;
  }();
  return init;
}

// Return the first hwloc object of the given type whose os_index
// matches 'index'.
hwloc_obj_t GetHWLocTypeIndex(hwloc_obj_type_t tp, int index) {
  hwloc_obj_t obj = nullptr;
  if (index >= 0) {
    while ((obj = hwloc_get_next_obj_by_type(hwloc_topology_handle, tp, obj)) !=
           nullptr) {
      if (obj->os_index == index) break;
    }
  }
  return obj;
}
}  // namespace
#endif  // defined(TACHYON_USE_NUMA)

bool NUMAEnabled() { return (NUMANumNodes() > 1); }

int NUMANumNodes() {
#if defined(TENSORFLOW_USE_NUMA)
  if (HaveHWLocTopology()) {
    int num_numanodes =
        hwloc_get_nbobjs_by_type(hwloc_topology_handle, HWLOC_OBJ_NUMANODE);
    return std::max(1, num_numanodes);
  } else {
    return 1;
  }
#else
  return 1;
#endif  // defined(TENSORFLOW_USE_NUMA)
}

void NUMASetThreadNodeAffinity(int node) {
#if defined(TACHYON_USE_NUMA)
  if (HaveHWLocTopology()) {
    // Find the corresponding NUMA node topology object.
    hwloc_obj_t obj = GetHWLocTypeIndex(HWLOC_OBJ_NUMANODE, node);
    if (obj) {
      hwloc_set_cpubind(hwloc_topology_handle, obj->cpuset,
                        HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);
    } else {
      LOG(ERROR) << "Could not find hwloc NUMA node " << node;
    }
  }
#endif  // defined(TACHYON_USE_NUMA)
}

int NUMAGetThreadNodeAffinity() {
  int node_index = kNUMANoAffinity;
#if defined(TACHYON_USE_NUMA)
  if (HaveHWLocTopology()) {
    hwloc_cpuset_t thread_cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(hwloc_topology_handle, thread_cpuset,
                      HWLOC_CPUBIND_THREAD);
    hwloc_obj_t obj = nullptr;
    // Return the first NUMA node whose cpuset is a (non-proper) superset of
    // that of the current thread.
    while ((obj = hwloc_get_next_obj_by_type(
                hwloc_topology_handle, HWLOC_OBJ_NUMANODE, obj)) != nullptr) {
      if (hwloc_bitmap_isincluded(thread_cpuset, obj->cpuset)) {
        node_index = obj->os_index;
        break;
      }
    }
    hwloc_bitmap_free(thread_cpuset);
  }
#endif  // defined(TACHYON_USE_NUMA)
  return node_index;
}

void* NUMAMalloc(int node, size_t size, int minimum_alignment) {
#if defined(TACHYON_USE_NUMA)
  if (HaveHWLocTopology()) {
    hwloc_obj_t numa_node = GetHWLocTypeIndex(HWLOC_OBJ_NUMANODE, node);
    if (numa_node) {
      return hwloc_alloc_membind(hwloc_topology_handle, size,
                                 numa_node->nodeset, HWLOC_MEMBIND_BIND,
                                 HWLOC_MEMBIND_BYNODESET);
    } else {
      LOG(ERROR) << "Failed to find hwloc NUMA node " << node;
    }
  }
#endif  // defined(TACHYON_USE_NUMA)
  return base::AlignedAlloc(size, minimum_alignment);
}

void NUMAFree(void* ptr, size_t size) {
#if defined(TACHYON_USE_NUMA)
  if (HaveHWLocTopology()) {
    hwloc_free(hwloc_topology_handle, ptr, size);
    return;
  }
#endif  // defined(TACHYON_USE_NUMA)
  base::AlignedFree(ptr);
}

int NUMAGetMemAffinity(const void* addr) {
  int node = kNUMANoAffinity;
#if defined(TACHYON_USE_NUMA)
  if (HaveHWLocTopology() && addr) {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    if (!hwloc_get_area_memlocation(hwloc_topology_handle, addr, 4, nodeset,
                                    HWLOC_MEMBIND_BYNODESET)) {
      hwloc_obj_t obj = nullptr;
      while ((obj = hwloc_get_next_obj_by_type(
                  hwloc_topology_handle, HWLOC_OBJ_NUMANODE, obj)) != nullptr) {
        if (hwloc_bitmap_isincluded(nodeset, obj->nodeset)) {
          node = obj->os_index;
          break;
        }
      }
      hwloc_bitmap_free(nodeset);
    } else {
      LOG(ERROR) << "Failed call to hwloc_get_area_memlocation.";
    }
  }
#endif  // defined(TACHYON_USE_NUMA)
  return node;
}

#endif
}  // namespace tachyon::device
