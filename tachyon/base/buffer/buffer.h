#ifndef TACHYON_BASE_BUFFER_BUFFER_H_
#define TACHYON_BASE_BUFFER_BUFFER_H_

#include "tachyon/base/buffer/read_only_buffer.h"

namespace tachyon::base {

class TACHYON_EXPORT Buffer : public ReadOnlyBuffer {
 public:
  Buffer() = default;
  Buffer(void* buffer, size_t buffer_len)
      : ReadOnlyBuffer(buffer, buffer_len) {}
  Buffer(const Buffer& other) = delete;
  Buffer& operator=(const Buffer& other) = delete;
  Buffer(Buffer&& other) = default;
  Buffer& operator=(Buffer&& other) = default;
  virtual ~Buffer() = default;

  // NOTE(chokobole): Due to the existence of a constant getter named |buffer()|
  // in the parent class, the name was chosen in snake case.
  using ReadOnlyBuffer::buffer;
  void* buffer() { return buffer_; }

  [[nodiscard]] bool Write(const uint8_t* ptr, size_t size) {
    return WriteAt(buffer_offset_, ptr, size);
  }

  template <typename T>
  [[nodiscard]] bool Write(const T& value) {
    return WriteAt(buffer_offset_, value);
  }

  [[nodiscard]] bool Write16BE(uint16_t value) {
    return Write16BEAt(buffer_offset_, value);
  }

  [[nodiscard]] bool Write16LE(uint16_t value) {
    return Write16LEAt(buffer_offset_, value);
  }

  [[nodiscard]] bool Write32BE(uint32_t value) {
    return Write32BEAt(buffer_offset_, value);
  }

  [[nodiscard]] bool Write32LE(uint32_t value) {
    return Write32LEAt(buffer_offset_, value);
  }

  [[nodiscard]] bool Write64BE(uint64_t value) {
    return Write64BEAt(buffer_offset_, value);
  }

  [[nodiscard]] bool Write64LE(uint64_t value) {
    return Write64LEAt(buffer_offset_, value);
  }

  template <typename T>
  [[nodiscard]] bool WriteMany(const T& value) {
    return Write(value);
  }

  template <typename T, typename... Args>
  [[nodiscard]] bool WriteMany(const T& value, const Args&... args) {
    if (!Write(value)) return false;
    return WriteMany(args...);
  }

  // Returns false when either
  // 1) |buffer_offset| + |size| overflows.
  // 2) if it is not growable and it tries to write more than
  // |buffer_len_|.
  [[nodiscard]] bool WriteAt(size_t buffer_offset, const uint8_t* ptr,
                             size_t size);

  [[nodiscard]] bool Write16BEAt(size_t buffer_offset, uint16_t ptr);
  [[nodiscard]] bool Write16LEAt(size_t buffer_offset, uint16_t ptr);
  [[nodiscard]] bool Write32BEAt(size_t buffer_offset, uint32_t ptr);
  [[nodiscard]] bool Write32LEAt(size_t buffer_offset, uint32_t ptr);
  [[nodiscard]] bool Write64BEAt(size_t buffer_offset, uint64_t ptr);
  [[nodiscard]] bool Write64LEAt(size_t buffer_offset, uint64_t ptr);

#define DEFINE_WRITE_AT(bytes, bits, type)                                \
  template <typename T,                                                   \
            std::enable_if_t<internal::IsBuiltinSerializable<T>::value && \
                             (sizeof(T) == bytes)>* = nullptr>            \
  [[nodiscard]] bool WriteAt(size_t buffer_offset, T value) {             \
    switch (endian_) {                                                    \
      case Endian::kBig:                                                  \
        return Write##bits##BEAt(buffer_offset, value);                   \
      case Endian::kLittle:                                               \
        return Write##bits##LEAt(buffer_offset, value);                   \
      case Endian::kNative:                                               \
        return WriteAt(buffer_offset,                                     \
                       reinterpret_cast<const uint8_t*>(&value), bytes);  \
    }                                                                     \
    NOTREACHED();                                                         \
    return false;                                                         \
  }

  DEFINE_WRITE_AT(2, 16, uint16_t)
  DEFINE_WRITE_AT(4, 32, uint32_t)
  DEFINE_WRITE_AT(8, 64, uint64_t)
#undef DEFINE_WRITE_AT

  template <typename T,
            std::enable_if_t<internal::IsBuiltinSerializable<T>::value &&
                             !((sizeof(T) == 2) || (sizeof(T) == 4) ||
                               (sizeof(T) == 8))>* = nullptr>
  [[nodiscard]] bool WriteAt(size_t buffer_offset, T value) {
    return WriteAt(buffer_offset, reinterpret_cast<const uint8_t*>(&value),
                   sizeof(T));
  }

  template <
      typename T,
      std::enable_if_t<internal::IsNonBuiltinSerializable<T>::value>* = nullptr>
  [[nodiscard]] bool WriteAt(size_t buffer_offset, const T& value) {
    buffer_offset_ = buffer_offset;
    return Copyable<T>::WriteTo(value, this);
  }

  template <typename T>
  [[nodiscard]] bool WriteManyAt(size_t buffer_offset, const T& value) {
    return WriteAt(buffer_offset, value);
  }

  template <typename T, typename... Args>
  [[nodiscard]] bool WriteManyAt(size_t buffer_offset, const T& value,
                                 const Args&... args) {
    if (!WriteAt(buffer_offset, value)) return false;
    return WriteManyAt(buffer_offset, args...);
  }

  [[nodiscard]] virtual bool Grow(size_t size) { return false; }
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_BUFFER_H_
