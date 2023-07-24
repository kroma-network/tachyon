#include "tachyon/math/finite_fields/goldilocks_prime/goldilocks_prime_field.h"

namespace tachyon {
namespace math {

// static
void GoldilocksConfig::Init() {
#if defined(TACHYON_GMP_BACKEND)
  GoldilocksGmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace math
}  // namespace tachyon
