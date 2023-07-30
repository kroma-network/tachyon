#include "tachyon/math/finite_fields/test/gf7.h"

namespace tachyon::math {

// static
void GF7Config::Init() {
#if defined(TACHYON_GMP_BACKEND)
  GF7Gmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace tachyon::math
