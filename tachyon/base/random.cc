#include "tachyon/base/random.h"

namespace tachyon::base {

absl::BitGen& GetAbslBitGen() {
  static absl::BitGen bitgen;
  return bitgen;
}

bool Bernoulli(double probability) {
  return absl::Bernoulli(GetAbslBitGen(), probability);
}

}  // namespace tachyon::base
