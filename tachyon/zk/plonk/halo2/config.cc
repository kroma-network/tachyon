#include "tachyon/zk/plonk/halo2/config.h"

namespace tachyon::zk::plonk::halo2 {

// static
Config& Config::Get() {
  static Config config;
  return config;
}

}  // namespace tachyon::zk::plonk::halo2
