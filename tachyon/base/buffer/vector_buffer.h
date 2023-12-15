#ifndef TACHYON_BASE_BUFFER_VECTOR_BUFFER_H_
#define TACHYON_BASE_BUFFER_VECTOR_BUFFER_H_

#include <utility>
#include <vector>

#include "tachyon/base/buffer/buffer.h"

namespace tachyon::base {

class TACHYON_EXPORT VectorBuffer : public Buffer {
 public:
  VectorBuffer() = default;
  VectorBuffer(const VectorBuffer& other) = delete;
  VectorBuffer& operator=(const VectorBuffer& other) = delete;
  VectorBuffer(VectorBuffer&& other)
      : Buffer(std::move(other)),
        owned_buffer_(std::move(other.owned_buffer_)) {}
  VectorBuffer& operator=(VectorBuffer&& other) {
    Buffer::operator=(std::move(other));
    owned_buffer_ = std::move(other.owned_buffer_);
    return *this;
  }
  ~VectorBuffer() override = default;

  const std::vector<char>& owned_buffer() const { return owned_buffer_; }

  std::vector<char>&& TakeOwnedBuffer() && { return std::move(owned_buffer_); }

  [[nodiscard]] bool Grow(size_t size) override {
    owned_buffer_.resize(size);
    buffer_ = owned_buffer_.data();
    buffer_len_ = owned_buffer_.size();
    return true;
  }

 protected:
  std::vector<char> owned_buffer_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_VECTOR_BUFFER_H_
