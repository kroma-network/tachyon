#include "tachyon/math/finite_fields/prime_field.h"

namespace tachyon {
namespace math {

// static
void GF7Config::Init() {
#if defined(TACHYON_GMP_BACKEND)
  GF7Gmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
#if TACHYON_CUDA
  GF7Cuda::Init();
#endif  // TACHYON_CUDA
}

}  // namespace math
}  // namespace tachyon
