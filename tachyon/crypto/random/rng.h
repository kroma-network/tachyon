#ifndef TACHYON_CRYPTO_RANDOM_RNG_H_
#define TACHYON_CRYPTO_RANDOM_RNG_H_

#include <stdint.h>

#include "tachyon/export.h"

namespace tachyon::crypto {

class TACHYON_EXPORT RNG {
 public:
  virtual ~RNG() = default;

  uint64_t NextUint64() {
    uint64_t lo = uint64_t{NextUint32()};
    uint64_t hi = uint64_t{NextUint32()};
    return (hi << 32 | lo);
  }

  virtual uint32_t NextUint32() = 0;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_RANDOM_RNG_H_
