#ifndef TACHYON_CRYPTO_HASHES_GROWABLE_BUFFER_H_
#define TACHYON_CRYPTO_HASHES_GROWABLE_BUFFER_H_

#include <vector>

#include "tachyon/crypto/hashes/buffer.h"

namespace tachyon::crypto {

class TACHYON_EXPORT GrowableBuffer : public Buffer {
 public:
  GrowableBuffer() = default;
  GrowableBuffer(const GrowableBuffer& other) = delete;
  GrowableBuffer& operator=(const GrowableBuffer& other) = delete;
  GrowableBuffer(GrowableBuffer&& other)
      : Buffer(std::move(other)),
        owned_buffer_(std::move(other.owned_buffer_)) {}
  GrowableBuffer& operator=(GrowableBuffer&& other) {
    owned_buffer_ = std::move(other.owned_buffer_);
    return *this;
  }
  ~GrowableBuffer() override = default;

  bool Grow(size_t size) override {
    owned_buffer_.resize(size);
    buffer_ = owned_buffer_.data();
    buffer_len_ = owned_buffer_.size();
    return true;
  }

 protected:
  std::vector<char> owned_buffer_;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_GROWABLE_BUFFER_H_
