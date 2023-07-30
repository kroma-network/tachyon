#include "tachyon/math/finite_fields/goldilocks_prime/goldilocks.h"

namespace tachyon::math {

// static
void GoldilocksConfig::Init() {
#if defined(TACHYON_GMP_BACKEND)
  GoldilocksGmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace tachyon::math
