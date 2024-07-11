#ifndef TACHYON_CRYPTO_RANDOM_BLOCK_RNG_H_
#define TACHYON_CRYPTO_RANDOM_BLOCK_RNG_H_

#include <stdint.h>

#include "tachyon/crypto/random/rng.h"

namespace tachyon::crypto {

template <typename Derived>
class BlockRNG : public RNG {
 public:
  constexpr static size_t kMaxIndex = Derived::kOutputSize / sizeof(uint32_t);

  BlockRNG() = default;
  explicit BlockRNG(size_t index) : index_(index) {}

  size_t index() const { return index_; }

  // RNG methods
  uint32_t NextUint32() override {
    auto derived = static_cast<Derived*>(this);
    if (index_ == kMaxIndex) {
      derived->Update();
      index_ = 0;
    }
    return derived->Get(index_++);
  }

 protected:
  size_t index_ = kMaxIndex;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_RANDOM_BLOCK_RNG_H_
