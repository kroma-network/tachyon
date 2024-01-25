#ifndef TACHYON_BASE_BUFFER_VECTOR_BUFFER_H_
#define TACHYON_BASE_BUFFER_VECTOR_BUFFER_H_

#include <utility>
#include <vector>

#include "tachyon/base/buffer/buffer.h"

namespace tachyon::base {

template <typename T>
class VectorBuffer : public Buffer {
 public:
  static_assert(sizeof(T) == 1);

  VectorBuffer() = default;
  explicit VectorBuffer(const std::vector<T>& owned_buffer)
      : owned_buffer_(owned_buffer) {
    UpdateBuffer();
  }
  explicit VectorBuffer(std::vector<T>&& owned_buffer)
      : owned_buffer_(std::move(owned_buffer)) {
    UpdateBuffer();
  }
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

  const std::vector<T>& owned_buffer() const { return owned_buffer_; }

  std::vector<T>&& TakeOwnedBuffer() && { return std::move(owned_buffer_); }

  [[nodiscard]] bool Grow(size_t size) override {
    owned_buffer_.resize(size);
    UpdateBuffer();
    return true;
  }

 protected:
  void UpdateBuffer() {
    buffer_ = owned_buffer_.data();
    buffer_len_ = owned_buffer_.size();
  }

  std::vector<T> owned_buffer_;
};

using CharVectorBuffer = VectorBuffer<char>;
using Uint8VectorBuffer = VectorBuffer<uint8_t>;

extern template class TACHYON_EXPORT VectorBuffer<char>;
extern template class TACHYON_EXPORT VectorBuffer<uint8_t>;

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_VECTOR_BUFFER_H_
