#include "tachyon/math/elliptic_curves/bls/bls12_381/fq.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

// static
void FqConfig::Init() {
  Fq::Init();
#if defined(TACHYON_GMP_BACKEND)
  FqGmp::Init();
#endif  // defined(TACHYON_GMP_BACKEND)
}

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon
