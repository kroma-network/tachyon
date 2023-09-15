#include "tachyon/crypto/hashes/buffer.h"

#include <string.h>

#include "tachyon/base/numerics/checked_math.h"

namespace tachyon::crypto {

bool Buffer::Write(const uint8_t* ptr, size_t size) {
  base::CheckedNumeric<size_t> len = buffer_offset_;
  size_t size_needed;
  if (!(len + size).AssignIfValid(&size_needed)) return false;
  if (size_needed > buffer_len_) {
    if (!Grow(size_needed)) return false;
  }
  memcpy(buffer_ + buffer_offset_, ptr, size);
  buffer_offset_ += size;
  return true;
}

}  // namespace tachyon::crypto
