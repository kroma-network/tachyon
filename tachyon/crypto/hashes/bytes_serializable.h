#ifndef TACHYON_CRYPTO_HASHES_BYTES_SERIALIZABLE_H_
#define TACHYON_CRYPTO_HASHES_BYTES_SERIALIZABLE_H_

#include <stdint.h>

#include <numeric>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/crypto/hashes/buffer.h"

namespace tachyon::crypto {

template <typename T, typename = void>
class BytesSerializable;

template <typename T>
class BytesSerializable<T, std::enable_if_t<std::is_trivially_copyable_v<T>>> {
 public:
  static bool ToBytes(const T& value, Buffer* buf) {
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(&value);
    return buf->Write(byte_ptr, sizeof(T));
  }

  static size_t GetSize(const T& value) { return sizeof(T); }

  static bool BatchToBytes(const absl::Span<const T>& values, Buffer* buf) {
    for (const T& value : values) {
      if (!BytesSerializable<T>::ToBytes(value, buf)) return false;
    }
    return true;
  }

  static size_t GetBatchSize(const absl::Span<const T>& values) {
    return values.size() * sizeof(T);
  }
};

template <typename T>
class BytesSerializable<std::vector<T>, void> {
 public:
  static bool ToBytes(const std::vector<T>& values, Buffer* buf) {
    if (!BytesSerializable<T>::ToBytes(values.size(), buf)) return false;
    for (const T& value : values) {
      if (!BytesSerializable<T>::ToBytes(value, buf)) return false;
    }
    return true;
  }

  static size_t GetSize(const std::vector<T>& values) {
    return std::accumulate(values.begin(), values.end(), size_t(0),
                           [](size_t total, const T& value) {
                             return total +
                                    BytesSerializable<T>::GetSize(value);
                           });
  }
};

template <typename T>
bool SerializeToBytes(const T& value, Buffer* buf) {
  return BytesSerializable<T>::ToBytes(value, buf);
}

template <typename T>
bool BatchSerializeToBytes(const std::vector<T>& value, Buffer* buf) {
  return BytesSerializable<T>::BatchToBytes(value, buf);
}

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_HASHES_BYTES_SERIALIZABLE_H_
