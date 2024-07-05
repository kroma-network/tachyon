#ifndef TACHYON_ZK_PLONK_HALO2_LS_TYPE_H_
#define TACHYON_ZK_PLONK_HALO2_LS_TYPE_H_

#include <stdint.h>

#include <string>

#include "absl/strings/substitute.h"

#include "tachyon/base/flag/flag_value_traits.h"

namespace tachyon {
namespace zk::plonk::halo2 {

// THE ORDER OF ELEMENTS ARE IMPORTANT!! DO NOT CHANGE!
// This order matches with the constants of c APIs.
// Pleas refer to tachyon/c/zk/plonk/halo2/constants.h for details.
enum class LSType : uint8_t {
  kHalo2,
  kLogDerivativeHalo2,
};

}  // namespace zk::plonk::halo2

namespace base {

template <>
class FlagValueTraits<zk::plonk::halo2::LSType> {
 public:
  static bool ParseValue(std::string_view input,
                         zk::plonk::halo2::LSType* value, std::string* reason) {
    if (input == "halo2") {
      *value = zk::plonk::halo2::LSType::kHalo2;
    } else if (input == "log_derivative_halo2") {
      *value = zk::plonk::halo2::LSType::kLogDerivativeHalo2;
    } else {
      *reason = absl::Substitute("Unknown ls type: $0", input);
      return false;
    }
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_LS_TYPE_H_
