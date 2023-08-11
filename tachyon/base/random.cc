#include "tachyon/base/random.h"

namespace tachyon::base {

absl::BitGen& GetAbslBitGen() {
  static absl::BitGen bitgen;
  return bitgen;
}

}  // namespace tachyon::base
