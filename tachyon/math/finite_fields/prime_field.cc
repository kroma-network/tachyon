#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {

// static
void GF7Config::Init() {
  GF7::Init();
#if defined(TACHYON_GMP_BACKEND)
  GF7Gmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace math
}  // namespace tachyon
