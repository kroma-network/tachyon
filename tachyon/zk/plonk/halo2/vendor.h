#ifndef TACHYON_ZK_PLONK_HALO2_VENDOR_H_
#define TACHYON_ZK_PLONK_HALO2_VENDOR_H_

#include <string_view>

#include "tachyon/export.h"

namespace tachyon::zk::plonk::halo2 {

enum class Vendor {
  kPSE,
  kScroll,
};

TACHYON_EXPORT std::string_view VendorToString(Vendor vendor);

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_VENDOR_H_
