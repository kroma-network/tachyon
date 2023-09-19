#ifndef TACHYON_BASE_BUFFER_BUFFER_H_
#define TACHYON_BASE_BUFFER_BUFFER_H_

#include <stddef.h>
#include <stdint.h>

#include <type_traits>
#include <utility>

#include "absl/base/internal/endian.h"

#include "tachyon/base/buffer/copyable_forward.h"
#include "tachyon/base/endian.h"
#include "tachyon/base/logging.h"

#include <iostream>

namespace tachyon::base {
namespace internal {

template <typename, typename = void>
struct IsBuiltinSerializable : std::false_type {};

template <typename T>
struct IsBuiltinSerializable<
    T, std::enable_if_t<std::is_fundamental_v<T> || std::is_enum_v<T>>>
    : std::true_type {};

template <typename, typename = void>
struct IsNonBuiltinSerializable : std::false_type {};

template <typename T>
struct IsNonBuiltinSerializable<
    T,
    std::enable_if_t<!IsBuiltinSerializable<T>::value && IsCopyable<T>::value>>
    : std::true_type {};

}  // namespace internal

// Buffer policy:
// It tries to write / read as much as possible.
// If errors occur during write or read, it is because the requested
// buffer offset combined with the size would overflow.
class TACHYON_EXPORT Buffer {
 public:
  Buffer() = default;
  Buffer(void* buffer, size_t buffer_len)
      : buffer_(buffer), buffer_offset_(0), buffer_len_(buffer_len) {}
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

  Endian endian() const { return endian_; }
  void set_endian(Endian endian) { endian_ = endian; }

  void* buffer() { return buffer_; }
  const void* buffer() const { return buffer_; }

  size_t buffer_offset() const { return buffer_offset_; }
  void set_buffer_offset(size_t buffer_offset) {
    buffer_offset_ = buffer_offset;
  }

  size_t buffer_len() const { return buffer_len_; }

  bool Done() const { return buffer_offset_ == buffer_len_; }

  bool Read(uint8_t* ptr, size_t size) const {
    return ReadAt(buffer_offset_, ptr, size);
  }

  template <typename T>
  bool Read(T* value) const {
    return ReadAt(buffer_offset_, value);
  }

  template <typename T>
  bool ReadMany(T* value) const {
    return Read(value);
  }

  template <typename T, typename... Args>
  bool ReadMany(T* value, Args*... args) const {
    if (!Read(value)) return false;
    return ReadMany(args...);
  }

  // Returns false either
  // 1) |buffer_offset| + |size| overflows.
  // 2) if it tries to read more than |buffer_len_|.
  bool ReadAt(size_t buffer_offset, uint8_t* ptr, size_t size) const;

  bool Read16BEAt(size_t buffer_offset, uint16_t* ptr) const;
  bool Read16LEAt(size_t buffer_offset, uint16_t* ptr) const;
  bool Read32BEAt(size_t buffer_offset, uint32_t* ptr) const;
  bool Read32LEAt(size_t buffer_offset, uint32_t* ptr) const;
  bool Read64BEAt(size_t buffer_offset, uint64_t* ptr) const;
  bool Read64LEAt(size_t buffer_offset, uint64_t* ptr) const;

#define DEFINE_READ_AT(bytes, bits, type)                                 \
  template <typename T,                                                   \
            std::enable_if_t<internal::IsBuiltinSerializable<T>::value && \
                             (sizeof(T) == bytes)>* = nullptr>            \
  bool ReadAt(size_t buffer_offset, T* value) const {                     \
    switch (endian_) {                                                    \
      case Endian::kBig:                                                  \
        return Read##bits##BEAt(buffer_offset,                            \
                                reinterpret_cast<type*>(value));          \
      case Endian::kLittle:                                               \
        return Read##bits##LEAt(buffer_offset,                            \
                                reinterpret_cast<type*>(value));          \
      case Endian::kNative:                                               \
        return ReadAt(buffer_offset, reinterpret_cast<uint8_t*>(value),   \
                      bytes);                                             \
    }                                                                     \
    NOTREACHED();                                                         \
    return false;                                                         \
  }

  DEFINE_READ_AT(2, 16, uint16_t)
  DEFINE_READ_AT(4, 32, uint32_t)
  DEFINE_READ_AT(8, 64, uint64_t)
#undef DEFINE_READ_AT

  template <typename T,
            std::enable_if_t<internal::IsBuiltinSerializable<T>::value &&
                             !((sizeof(T) == 2) || (sizeof(T) == 4) ||
                               (sizeof(T) == 8))>* = nullptr>
  bool ReadAt(size_t buffer_offset, T* value) const {
    return ReadAt(buffer_offset, reinterpret_cast<uint8_t*>(value), sizeof(T));
  }

  template <
      typename T,
      std::enable_if_t<internal::IsNonBuiltinSerializable<T>::value>* = nullptr>
  bool ReadAt(size_t buffer_offset, T* value) const {
    buffer_offset_ = buffer_offset;
    return Copyable<T>::ReadFrom(*this, value);
  }

  template <typename T>
  bool ReadManyAt(size_t buffer_offset, T* value) const {
    return ReadAt(buffer_offset, value);
  }

  template <typename T, typename... Args>
  bool ReadManyAt(size_t buffer_offset, T* value, Args*... args) const {
    if (!ReadAt(buffer_offset, value)) return false;
    return ReadManyAt(buffer_offset, args...);
  }

  bool Write(const uint8_t* ptr, size_t size) {
    return WriteAt(buffer_offset_, ptr, size);
  }

  template <typename T>
  bool Write(const T& value) {
    return WriteAt(buffer_offset_, value);
  }

  template <typename T>
  bool WriteMany(const T& value) {
    return Write(value);
  }

  template <typename T, typename... Args>
  bool WriteMany(const T& value, const Args&... args) {
    if (!Write(value)) return false;
    return WriteMany(args...);
  }

  // Returns false either
  // 1) |buffer_offset| + |size| overflows.
  // 2) if it is not growable and it tries to write more than
  // |buffer_len_|.
  bool WriteAt(size_t buffer_offset, const uint8_t* ptr, size_t size);

  bool Write16BEAt(size_t buffer_offset, uint16_t ptr);
  bool Write16LEAt(size_t buffer_offset, uint16_t ptr);
  bool Write32BEAt(size_t buffer_offset, uint32_t ptr);
  bool Write32LEAt(size_t buffer_offset, uint32_t ptr);
  bool Write64BEAt(size_t buffer_offset, uint64_t ptr);
  bool Write64LEAt(size_t buffer_offset, uint64_t ptr);

#define DEFINE_WRITE_AT(bytes, bits, type)                                \
  template <typename T,                                                   \
            std::enable_if_t<internal::IsBuiltinSerializable<T>::value && \
                             (sizeof(T) == bytes)>* = nullptr>            \
  bool WriteAt(size_t buffer_offset, T value) {                           \
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
  bool WriteAt(size_t buffer_offset, T value) {
    return WriteAt(buffer_offset, reinterpret_cast<const uint8_t*>(&value),
                   sizeof(T));
  }

  template <
      typename T,
      std::enable_if_t<internal::IsNonBuiltinSerializable<T>::value>* = nullptr>
  bool WriteAt(size_t buffer_offset, const T& value) {
    buffer_offset_ = buffer_offset;
    return Copyable<T>::WriteTo(value, this);
  }

  template <typename T>
  bool WriteManyAt(size_t buffer_offset, const T& value) {
    return WriteAt(buffer_offset, value);
  }

  template <typename T, typename... Args>
  bool WriteManyAt(size_t buffer_offset, const T& value, const Args&... args) {
    if (!WriteAt(buffer_offset, value)) return false;
    return WriteManyAt(buffer_offset, args...);
  }

  virtual bool Grow(size_t size) { return false; }

 protected:
  Endian endian_ = Endian::kNative;

  void* buffer_ = nullptr;
  mutable size_t buffer_offset_ = 0;
  size_t buffer_len_ = 0;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_BUFFER_H_
