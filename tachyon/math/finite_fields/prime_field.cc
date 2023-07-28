#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon::math {

// static
void GF7Config::Init() {
#if defined(TACHYON_GMP_BACKEND)
  GF7Gmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace tachyon::math
