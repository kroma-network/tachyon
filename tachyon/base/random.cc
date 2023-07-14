#include "tachyon/base/random.h"

namespace tachyon {
namespace base {

absl::BitGen& GetAbslBitGen() {
  static absl::BitGen bitgen;
  return bitgen;
}

}  // namespace base
}  // namespace tachyon
