#ifndef TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_CONFIG_H_
#define TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_CONFIG_H_

#include <stddef.h>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/export.h"

namespace tachyon {
namespace crypto {

struct TACHYON_EXPORT SpongeConfig {
  // The rate (in terms of number of field elements).
  // See https://iacr.org/archive/eurocrypt2008/49650180/49650180.pdf
  size_t rate = 0;

  // The capacity (in terms of number of field elements).
  size_t capacity = 0;

  SpongeConfig() = default;
  SpongeConfig(size_t rate, size_t capacity) : rate(rate), capacity(capacity) {}

  bool operator==(const SpongeConfig& other) const {
    return rate == other.rate && capacity == other.capacity;
  }
  bool operator!=(const SpongeConfig& other) const {
    return !operator==(other);
  }
};

}  // namespace crypto

namespace base {

template <>
class Copyable<crypto::SpongeConfig> {
 public:
  static bool WriteTo(const crypto::SpongeConfig& config, Buffer* buffer) {
    return buffer->WriteMany(config.rate, config.capacity);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       crypto::SpongeConfig* config) {
    size_t rate;
    size_t capacity;
    if (!buffer.ReadMany(&rate, &capacity)) {
      return false;
    }

    *config = {rate, capacity};
    return true;
  }

  static size_t EstimateSize(const crypto::SpongeConfig& config) {
    return base::EstimateSize(config.rate, config.capacity);
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_HASHES_SPONGE_SPONGE_CONFIG_H_
