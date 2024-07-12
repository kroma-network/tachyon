#ifndef TACHYON_CRYPTO_RANDOM_RNG_TYPE_H_
#define TACHYON_CRYPTO_RANDOM_RNG_TYPE_H_

#include <stdint.h>

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/base/flag/flag_value_traits.h"

namespace tachyon {
namespace crypto {

// THE ORDER OF ELEMENTS ARE IMPORTANT!! DO NOT CHANGE!
// This order matches with the constants of c APIs.
// Pleas refer to tachyon/c/crypto/random.h for details.
enum class RNGType : uint8_t {
  kXORShift,
  kChaCha20,
};

}  // namespace crypto

namespace base {

template <>
class FlagValueTraits<crypto::RNGType> {
 public:
  static bool ParseValue(std::string_view input, crypto::RNGType* value,
                         std::string* reason) {
    if (input == "xor_shift") {
      *value = crypto::RNGType::kXORShift;
    } else if (input == "cha_cha20") {
      *value = crypto::RNGType::kChaCha20;
    } else {
      *reason = absl::Substitute("Unknown rng type: $0", input);
      return false;
    }
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_RANDOM_RNG_TYPE_H_
