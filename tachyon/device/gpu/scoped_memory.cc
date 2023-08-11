#include "tachyon/device/gpu/scoped_memory.h"

#include "tachyon/base/logging.h"

namespace tachyon::device::gpu {

std::string MemoryTypeToString(MemoryType type) {
  switch (type) {
    case MemoryType::kDevice:
      return "Device";
    case MemoryType::kPageLocked:
      return "PageLocked";
#if TACHYON_CUDA
    case MemoryType::kUnified:
      return "Unified";
#endif  // TACHYON_CUDA
  }
  NOTREACHED();
  return "";
}

std::ostream& operator<<(std::ostream& os, MemoryType type) {
  return os << MemoryTypeToString(type);
}

}  // namespace tachyon::device::gpu
