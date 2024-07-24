#include "tachyon/zk/plonk/halo2/vendor.h"

#include "tachyon/base/logging.h"

namespace tachyon::zk::plonk::halo2 {

std::string_view VendorToString(Vendor vendor) {
  switch (vendor) {
    case Vendor::kScroll:
      return "scroll";
  }
  NOTREACHED();
  return "";
}

}  // namespace tachyon::zk::plonk::halo2
