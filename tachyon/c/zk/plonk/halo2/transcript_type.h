#ifndef TACHYON_C_ZK_PLONK_HALO2_TRANSCRIPT_TYPE_H_
#define TACHYON_C_ZK_PLONK_HALO2_TRANSCRIPT_TYPE_H_

#include <stdint.h>

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/base/flag/flag_value_traits.h"

namespace tachyon {
namespace c::zk::plonk::halo2 {

enum class TranscriptType : uint8_t {
  kBlake2b,
};

}  // namespace c::zk::plonk::halo2

namespace base {

template <>
class FlagValueTraits<c::zk::plonk::halo2::TranscriptType> {
 public:
  static bool ParseValue(std::string_view input,
                         c::zk::plonk::halo2::TranscriptType* value,
                         std::string* reason) {
    if (input == "blake2b") {
      *value = c::zk::plonk::halo2::TranscriptType::kBlake2b;
    } else {
      *reason = absl::Substitute("Unknown test set: $0", input);
      return false;
    }
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_C_ZK_PLONK_HALO2_TRANSCRIPT_TYPE_H_
