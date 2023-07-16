#include "tachyon/math/elliptic_curves/bn/bn254/fq.h"

#include "absl/base/call_once.h"

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/gmp/gmp_util.h"

namespace tachyon {
namespace math {
namespace bn254 {

// static
void FqConfig::Init() {
  static absl::once_flag once;
  absl::call_once(once, []() {
#if defined(TACHYON_GMP_BACKEND)
    mpz_class modulus;
    // Hex: 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
    gmp::MustParseIntoMpz(
        "2188824287183927522224640574525727508869631115729782366268903789464522"
        "6208583",
        10, &modulus);
    Modulus() = Fq(modulus, true);
#endif
  });
}

// static
Fq& FqConfig::Modulus() {
  static base::NoDestructor<Fq> modulus;
  return *modulus;
}

}  // namespace bn254
}  // namespace math
}  // namespace tachyon
