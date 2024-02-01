#ifndef TACHYON_RS_BASE_RUST_VEC_COPYABLE_H_
#define TACHYON_RS_BASE_RUST_VEC_COPYABLE_H_

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "absl/strings/substitute.h"
#include "absl/types/span.h"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_number_conversions.h"
#include "tachyon/export.h"

namespace tachyon {
namespace rs {

struct TACHYON_EXPORT RustVec {
  uintptr_t ptr;
  size_t capacity;
  size_t length;

  template <typename T>
  absl::Span<T> ToSpan() {
    return absl::Span(reinterpret_cast<T*>(ptr), length);
  }

  template <typename T>
  std::vector<T> ToVec() {
    return std::vector<T>(reinterpret_cast<T*>(ptr),
                          reinterpret_cast<T*>(ptr) + length);
  }

  std::string ToString() const {
    return absl::Substitute("{ptr: $0, capacity: $1, length: $2}",
                            base::HexToString(ptr), capacity, length);
  }
};

}  // namespace rs

namespace base {

template <>
class Copyable<rs::RustVec> {
 public:
  static bool WriteTo(const rs::RustVec& rust_vec, Buffer* buffer) {
    NOTREACHED();
    return false;
  }

  static bool ReadFrom(const Buffer& buffer, rs::RustVec* rust_vec) {
    uintptr_t ptr;
    size_t capacity;
    size_t length;
    if (!buffer.ReadMany(&ptr, &capacity, &length)) return false;
    *rust_vec = {ptr, capacity, length};
    return true;
  }

  static size_t EstimateSize(const rs::RustVec& rust_vec) {
    return sizeof(rust_vec.ptr) + sizeof(rust_vec.capacity) +
           sizeof(rust_vec.length);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_RS_BASE_RUST_VEC_COPYABLE_H_
