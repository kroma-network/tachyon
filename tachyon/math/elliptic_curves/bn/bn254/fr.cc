#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon {
namespace math {
namespace bn254 {

// static
void FrConfig::Init() {
#if defined(TACHYON_GMP_BACKEND)
  FrGmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace bn254
}  // namespace math
}  // namespace tachyon
