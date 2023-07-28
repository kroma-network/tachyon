#include "tachyon/math/elliptic_curves/bls/bls12_381/fr.h"

namespace tachyon::math {
namespace bls12_381 {

// static
void FrConfig::Init() {
#if defined(TACHYON_GMP_BACKEND)
  FrGmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace bls12_381
}  // namespace tachyon::math
