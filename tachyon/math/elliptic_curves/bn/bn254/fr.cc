#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

#include "absl/base/call_once.h"

#include "tachyon/math/base/gmp_util.h"

namespace tachyon {
namespace math {
namespace bn254 {

// static
void Fr::Init() {
  static absl::once_flag once;
  absl::call_once(once, []() {
#if defined(TACHYON_GMP_BACKEND)
    gmp::MustParseIntoMpz(
        "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
        16, &RawModulus());
#endif
  });
}

}  // namespace bn254
}  // namespace math
}  // namespace tachyon
