#ifndef TACHYON_CRYPTO_RANDOM_RNG_H_
#define TACHYON_CRYPTO_RANDOM_RNG_H_

#include <stdint.h>

namespace tachyon::crypto {

template <typename Derived>
class RNG {
 public:
  uint64_t NextUint64() {
    Derived* derived = static_cast<Derived*>(this);
    uint64_t lo = uint64_t{derived->NextUint32()};
    uint64_t hi = uint64_t{derived->NextUint32()};
    return (hi << 32 | lo);
  }
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_RANDOM_RNG_H_
