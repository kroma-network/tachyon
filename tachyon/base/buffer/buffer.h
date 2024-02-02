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

  [[nodiscard]] bool Done() const { return buffer_offset_ == buffer_len_; }

  // Returns false either
  // 1) |buffer_offset| + |size| overflows.
  // 2) if it tries to read more than |buffer_len_|.
  [[nodiscard]] bool ReadAt(size_t buffer_offset, uint8_t* ptr,
                            size_t size) const;

  template <
      typename Ptr, typename T = std::remove_pointer_t<Ptr>,
      std::enable_if_t<std::is_pointer_v<Ptr> &&
                       internal::IsBuiltinSerializable<T>::value>* = nullptr>
  [[nodiscard]] bool ReadAt(size_t buffer_offset, Ptr ptr) const {
    switch (endian_) {
      case Endian::kBig:
        if constexpr (sizeof(T) == 8) {
          return Read64BEAt(buffer_offset, reinterpret_cast<uint64_t*>(ptr));
        } else if constexpr (sizeof(T) == 4) {
          return Read32BEAt(buffer_offset, reinterpret_cast<uint32_t*>(ptr));
        } else if constexpr (sizeof(T) == 2) {
          return Read16BEAt(buffer_offset, reinterpret_cast<uint16_t*>(ptr));
        }
      case Endian::kLittle:
        if constexpr (sizeof(T) == 8) {
          return Read64LEAt(buffer_offset, reinterpret_cast<uint64_t*>(ptr));
        } else if constexpr (sizeof(T) == 4) {
          return Read32LEAt(buffer_offset, reinterpret_cast<uint32_t*>(ptr));
        } else if constexpr (sizeof(T) == 2) {
          return Read16LEAt(buffer_offset, reinterpret_cast<uint16_t*>(ptr));
        }
      case Endian::kNative:
        return ReadAt(buffer_offset, reinterpret_cast<uint8_t*>(ptr),
                      sizeof(T));
    }
    NOTREACHED();
    return false;
  }

  [[nodiscard]] bool Read16BEAt(size_t buffer_offset, uint16_t* ptr) const;
  [[nodiscard]] bool Read16LEAt(size_t buffer_offset, uint16_t* ptr) const;
  [[nodiscard]] bool Read32BEAt(size_t buffer_offset, uint32_t* ptr) const;
  [[nodiscard]] bool Read32LEAt(size_t buffer_offset, uint32_t* ptr) const;
  [[nodiscard]] bool Read64BEAt(size_t buffer_offset, uint64_t* ptr) const;
  [[nodiscard]] bool Read64LEAt(size_t buffer_offset, uint64_t* ptr) const;

  template <
      typename T,
      std::enable_if_t<internal::IsNonBuiltinSerializable<T>::value>* = nullptr>
  [[nodiscard]] bool ReadAt(size_t buffer_offset, T* value) const {
    buffer_offset_ = buffer_offset;
    return Copyable<T>::ReadFrom(*this, value);
  }

  template <typename T, size_t N>
  [[nodiscard]] bool ReadAt(size_t buffer_offset, T (&array)[N]) const {
    buffer_offset_ = buffer_offset;
    for (size_t i = 0; i < N; ++i) {
      if (!Read(&array[i])) return false;
    }
    return true;
  }

  [[nodiscard]] bool Read(uint8_t* ptr, size_t size) const {
    return ReadAt(buffer_offset_, ptr, size);
  }

  template <typename T>
  [[nodiscard]] bool Read(T&& value) const {
    return ReadAt(buffer_offset_, std::forward<T>(value));
  }

  [[nodiscard]] bool Read16BE(uint16_t* ptr) const {
    return Read16BEAt(buffer_offset_, ptr);
  }

  [[nodiscard]] bool Read16LE(uint16_t* ptr) const {
    return Read16LEAt(buffer_offset_, ptr);
  }

  [[nodiscard]] bool Read32BE(uint32_t* ptr) const {
    return Read32BEAt(buffer_offset_, ptr);
  }

  [[nodiscard]] bool Read32LE(uint32_t* ptr) const {
    return Read32LEAt(buffer_offset_, ptr);
  }

  [[nodiscard]] bool Read64BE(uint64_t* ptr) const {
    return Read64BEAt(buffer_offset_, ptr);
  }

  [[nodiscard]] bool Read64LE(uint64_t* ptr) const {
    return Read64LEAt(buffer_offset_, ptr);
  }

  template <typename T>
  [[nodiscard]] bool ReadMany(T&& value) const {
    return Read(std::forward<T>(value));
  }

  template <typename T, typename... Args>
  [[nodiscard]] bool ReadMany(T&& value, Args&&... args) const {
    if (!Read(std::forward<T>(value))) return false;
    return ReadMany(std::forward<Args>(args)...);
  }

  template <typename T, size_t N>
  [[nodiscard]] bool ReadManyAt(size_t buffer_offset, T&& value) const {
    return ReadAt(buffer_offset, std::forward<T>(value));
  }

  template <typename T, typename... Args>
  [[nodiscard]] bool ReadManyAt(size_t buffer_offset, T&& value,
                                Args&&... args) const {
    if (!ReadAt(buffer_offset, std::forward<T>(value))) return false;
    return ReadManyAt(buffer_offset, std::forward<Args>(args)...);
  }

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

  // Returns false either
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

 protected:
  Endian endian_ = Endian::kNative;

  void* buffer_ = nullptr;
  mutable size_t buffer_offset_ = 0;
  size_t buffer_len_ = 0;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_BUFFER_H_
