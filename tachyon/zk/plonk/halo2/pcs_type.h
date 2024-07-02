#ifndef TACHYON_ZK_PLONK_HALO2_PCS_TYPE_H_
#define TACHYON_ZK_PLONK_HALO2_PCS_TYPE_H_

#include <stdint.h>

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/base/flag/flag_value_traits.h"

namespace tachyon {
namespace zk::plonk::halo2 {

// THE ORDER OF ELEMENTS ARE IMPORTANT!! DO NOT CHANGE!
// This order matches with the constants of c APIs.
// Pleas refer to tachyon/c/zk/plonk/halo2/constants.h for details.
enum class PCSType : uint8_t {
  kGWC,
  kSHPlonk,
};

}  // namespace zk::plonk::halo2

namespace base {

template <>
class FlagValueTraits<zk::plonk::halo2::PCSType> {
 public:
  static bool ParseValue(std::string_view input,
                         zk::plonk::halo2::PCSType* value,
                         std::string* reason) {
    if (input == "gwc") {
      *value = zk::plonk::halo2::PCSType::kGWC;
    } else if (input == "shplonk") {
      *value = zk::plonk::halo2::PCSType::kSHPlonk;
    } else {
      *reason = absl::Substitute("Unknown pcs type: $0", input);
      return false;
    }
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_PCS_TYPE_H_
