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
    owned_buffer_ = std::move(other.owned_buffer_);
    return *this;
  }
  ~StringBuffer() override = default;

  bool Grow(size_t size) override {
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
