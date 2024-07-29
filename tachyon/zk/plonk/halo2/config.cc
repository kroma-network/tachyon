#include "tachyon/zk/plonk/halo2/config.h"

namespace tachyon::zk::plonk::halo2 {

Config& GetConfig() {
  static Config config;
  return config;
}

}  // namespace tachyon::zk::plonk::halo2
