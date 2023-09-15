#ifndef TACHYON_CRYPTO_HASHES_BUFFER_H_
#define TACHYON_CRYPTO_HASHES_BUFFER_H_

#include <stddef.h>

#include <utility>

#include "tachyon/export.h"

namespace tachyon::crypto {

class TACHYON_EXPORT Buffer {
 public:
  Buffer() = default;
  Buffer(const Buffer& other) = delete;
  Buffer& operator=(const Buffer& other) = delete;
  Buffer(Buffer&& other)
      : buffer_(std::exchange(other.buffer_, nullptr)),
        buffer_offset_(std::exchange(other.buffer_offset_, 0)),
        buffer_len_(std::exchange(other.buffer_len_, 0)) {}
  Buffer& operator=(Buffer&& other) {
    buffer_ = std::exchange(other.buffer_, nullptr);
    buffer_offset_ = std::exchange(other.buffer_offset_, 0);
    buffer_len_ = std::exchange(other.buffer_len_, 0);
    return *this;
  }
  virtual ~Buffer() = default;

  char* buffer() { return buffer_; }
  const char* buffer() const { return buffer_; }
  size_t buffer_offset() const { return buffer_offset_; }
  void set_buffer_offset(size_t buffer_offset) {
    buffer_offset_ = buffer_offset;
  }

  bool Write(const uint8_t* ptr, size_t size);

  virtual bool Grow(size_t size) { return false; }

 protected:
  char* buffer_ = nullptr;
  size_t buffer_offset_ = 0;
  size_t buffer_len_ = 0;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_BUFFER_H_
