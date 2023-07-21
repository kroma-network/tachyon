#include "tachyon/math/elliptic_curves/secp/secp256k1/fq.h"

namespace tachyon {
namespace math {
namespace secp256k1 {

// static
void FqConfig::Init() {
#if defined(TACHYON_GMP_BACKEND)
  FqGmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace secp256k1
}  // namespace math
}  // namespace tachyon
