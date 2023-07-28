#include "tachyon/math/elliptic_curves/bls/bls12_381/fq.h"

namespace tachyon::math {
namespace bls12_381 {

// static
void FqConfig::Init() {
#if defined(TACHYON_GMP_BACKEND)
  FqGmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace bls12_381
}  // namespace tachyon::math
