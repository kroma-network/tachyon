#include "tachyon/math/elliptic_curves/secp/secp256k1/fr.h"

namespace tachyon::math {
namespace secp256k1 {

// static
void FrConfig::Init() {
#if defined(TACHYON_GMP_BACKEND)
  FrGmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace secp256k1
}  // namespace tachyon::math
