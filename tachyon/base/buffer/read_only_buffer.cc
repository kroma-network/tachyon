#include "tachyon/base/buffer/read_only_buffer.h"

#include <string.h>

#include "tachyon/base/numerics/checked_math.h"

namespace tachyon::base {

bool ReadOnlyBuffer::ReadAt(size_t buffer_offset, uint8_t* ptr,
                            size_t size) const {
  base::CheckedNumeric<size_t> len = buffer_offset;
  size_t size_needed;
  if (!(len + size).AssignIfValid(&size_needed)) return false;
  if (size_needed > buffer_len_) {
    return false;
  }
  const char* buffer = reinterpret_cast<const char*>(buffer_);
  memcpy(ptr, &buffer[buffer_offset], size);
  buffer_offset_ = buffer_offset + size;
  return true;
}

#define READ_BE_AT(bytes, bits, type)                                    \
  bool ReadOnlyBuffer::Read##bits##BEAt(size_t buffer_offset, type* ptr) \
      const {                                                            \
    base::CheckedNumeric<size_t> len = buffer_offset;                    \
    size_t size_needed;                                                  \
    if (!(len + bytes).AssignIfValid(&size_needed)) return false;        \
    if (size_needed > buffer_len_) {                                     \
      return false;                                                      \
    }                                                                    \
    const char* buffer = reinterpret_cast<char*>(buffer_);               \
    type value = absl::big_endian::Load##bits(&buffer[buffer_offset]);   \
    memcpy(ptr, &value, bytes);                                          \
    buffer_offset_ = buffer_offset + bytes;                              \
    return true;                                                         \
  }

READ_BE_AT(2, 16, uint16_t)
READ_BE_AT(4, 32, uint32_t)
READ_BE_AT(8, 64, uint64_t)

#undef READ_BE_AT

#define READ_LE_AT(bytes, bits, type)                                     \
  bool ReadOnlyBuffer::Read##bits##LEAt(size_t buffer_offset, type* ptr)  \
      const {                                                             \
    base::CheckedNumeric<size_t> len = buffer_offset;                     \
    size_t size_needed;                                                   \
    if (!(len + bytes).AssignIfValid(&size_needed)) return false;         \
    if (size_needed > buffer_len_) {                                      \
      return false;                                                       \
    }                                                                     \
    const char* buffer = reinterpret_cast<const char*>(buffer_);          \
    type value = absl::little_endian::Load##bits(&buffer[buffer_offset]); \
    memcpy(ptr, &value, bytes);                                           \
    buffer_offset_ = buffer_offset + bytes;                               \
    return true;                                                          \
  }

READ_LE_AT(2, 16, uint16_t)
READ_LE_AT(4, 32, uint32_t)
READ_LE_AT(8, 64, uint64_t)

#undef READ_LE_AT

}  // namespace tachyon::base
