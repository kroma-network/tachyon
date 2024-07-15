#ifndef TACHYON_CRYPTO_RANDOM_RNG_H_
#define TACHYON_CRYPTO_RANDOM_RNG_H_

#include <stdint.h>

#include "absl/types/span.h"

#include "tachyon/base/buffer/buffer.h"
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

  virtual void SetRandomSeed() = 0;

  // Return false if the length of the |seed| exceeds the expected seed size.
  [[nodiscard]] virtual bool SetSeed(absl::Span<const uint8_t> seed) = 0;

  virtual uint32_t NextUint32() = 0;

  // Return false if it fails to read the state from the |buffer|.
  [[nodiscard]] virtual bool ReadFromBuffer(
      const base::ReadOnlyBuffer& buffer) = 0;

  // Return false if it fails to write the state to the |buffer|.
  [[nodiscard]] virtual bool WriteToBuffer(base::Buffer& buffer) const = 0;
};

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_RANDOM_RNG_H_
