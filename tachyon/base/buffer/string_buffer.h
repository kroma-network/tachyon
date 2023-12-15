#ifndef TACHYON_BASE_BUFFER_STRING_BUFFER_H_
#define TACHYON_BASE_BUFFER_STRING_BUFFER_H_

#include <string>
#include <utility>

#include "tachyon/base/buffer/buffer.h"

namespace tachyon::base {

class TACHYON_EXPORT StringBuffer : public Buffer {
 public:
  StringBuffer() = default;
  StringBuffer(const StringBuffer& other) = delete;
  StringBuffer& operator=(const StringBuffer& other) = delete;
  StringBuffer(StringBuffer&& other)
      : Buffer(std::move(other)),
        owned_buffer_(std::move(other.owned_buffer_)) {}
  StringBuffer& operator=(StringBuffer&& other) {
    Buffer::operator=(std::move(other));
    owned_buffer_ = std::move(other.owned_buffer_);
    return *this;
  }
  ~StringBuffer() override = default;

  const std::string& owned_buffer() const { return owned_buffer_; }

  std::string&& TakeOwnedBuffer() && { return std::move(owned_buffer_); }

  [[nodiscard]] bool Grow(size_t size) override {
    owned_buffer_.resize(size);
    buffer_ = owned_buffer_.data();
    buffer_len_ = owned_buffer_.size();
    return true;
  }

 protected:
  std::string owned_buffer_;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_STRING_BUFFER_H_
