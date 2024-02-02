#ifndef TACHYON_BASE_BUFFER_READ_ONLY_BUFFER_H_
#define TACHYON_BASE_BUFFER_READ_ONLY_BUFFER_H_

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

class TACHYON_EXPORT ReadOnlyBuffer {
 public:
  ReadOnlyBuffer() = default;
  ReadOnlyBuffer(const void* buffer, size_t buffer_len)
      : buffer_(const_cast<void*>(buffer)),
        buffer_offset_(0),
        buffer_len_(buffer_len) {}
  ReadOnlyBuffer(const ReadOnlyBuffer& other) = delete;
  ReadOnlyBuffer& operator=(const ReadOnlyBuffer& other) = delete;
  ReadOnlyBuffer(ReadOnlyBuffer&& other)
      : buffer_(std::exchange(other.buffer_, nullptr)),
        buffer_offset_(std::exchange(other.buffer_offset_, 0)),
        buffer_len_(std::exchange(other.buffer_len_, 0)) {}
  ReadOnlyBuffer& operator=(ReadOnlyBuffer&& other) {
    buffer_ = std::exchange(other.buffer_, nullptr);
    buffer_offset_ = std::exchange(other.buffer_offset_, 0);
    buffer_len_ = std::exchange(other.buffer_len_, 0);
    return *this;
  }
  ~ReadOnlyBuffer() = default;

  Endian endian() const { return endian_; }
  // NOTE(chokobole): This is changed to const method to ensure
  // |const ReadOnlyBuffer&| calls this.
  void set_endian(Endian endian) const { endian_ = endian; }

  const void* buffer() const { return buffer_; }

  size_t buffer_offset() const { return buffer_offset_; }
  void set_buffer_offset(size_t buffer_offset) {
    buffer_offset_ = buffer_offset;
  }

  size_t buffer_len() const { return buffer_len_; }

  [[nodiscard]] bool Done() const { return buffer_offset_ == buffer_len_; }

  // Returns false when either
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

 protected:
  mutable Endian endian_ = Endian::kNative;

  void* buffer_ = nullptr;
  mutable size_t buffer_offset_ = 0;
  size_t buffer_len_ = 0;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_BUFFER_READ_ONLY_BUFFER_H_
