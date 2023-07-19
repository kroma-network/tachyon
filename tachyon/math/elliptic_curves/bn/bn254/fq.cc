#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"

namespace tachyon {
namespace math {
namespace bn254 {

// static
void FqConfig::Init() {
#if defined(TACHYON_GMP_BACKEND)
  FqGmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace bn254
}  // namespace math
}  // namespace tachyon
