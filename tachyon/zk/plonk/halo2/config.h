#ifndef TACHYON_ZK_PLONK_HALO2_CONFIG_H_
#define TACHYON_ZK_PLONK_HALO2_CONFIG_H_

#include <stdint.h>

#include "tachyon/export.h"
#include "tachyon/zk/plonk/halo2/vendor.h"

namespace tachyon::zk::plonk::halo2 {

struct TACHYON_EXPORT Config {
  // By default, halo2 behaves like scroll halo2 v1.1.
  Vendor vendor = Vendor::kScroll;
  uint32_t version = 10100;

  static Config& Get();
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_CONFIG_H_
