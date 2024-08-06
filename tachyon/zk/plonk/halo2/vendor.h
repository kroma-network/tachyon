#ifndef TACHYON_ZK_PLONK_HALO2_VENDOR_H_
#define TACHYON_ZK_PLONK_HALO2_VENDOR_H_

#include <string>
#include <string_view>

#include "absl/strings/substitute.h"

#include "tachyon/base/flag/flag_value_traits.h"
#include "tachyon/export.h"

namespace tachyon {
namespace zk::plonk::halo2 {

// THE ORDER OF ELEMENTS ARE IMPORTANT!! DO NOT CHANGE!
// This order matches with the constants of c APIs.
// Pleas refer to tachyon/c/zk/plonk/halo2/constants.h for details.
enum class Vendor {
  kPSE,
  kScroll,
};

TACHYON_EXPORT std::string_view VendorToString(Vendor vendor);

}  // namespace zk::plonk::halo2

namespace base {

template <>
class FlagValueTraits<zk::plonk::halo2::Vendor> {
 public:
  static bool ParseValue(std::string_view input,
                         zk::plonk::halo2::Vendor* value, std::string* reason) {
    if (input == "pse") {
      *value = zk::plonk::halo2::Vendor::kPSE;
    } else if (input == "scroll") {
      *value = zk::plonk::halo2::Vendor::kScroll;
    } else {
      *reason = absl::Substitute("Unknown vendor: $0", input);
      return false;
    }
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_VENDOR_H_
