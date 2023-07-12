#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

#include "absl/base/call_once.h"

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/gmp_util.h"

namespace tachyon {
namespace math {
namespace bn254 {

// static
void FrConfig::Init() {
  static absl::once_flag once;
  absl::call_once(once, []() {
#if defined(TACHYON_GMP_BACKEND)
    mpz_class modulus;
    // Hex: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
    gmp::MustParseIntoMpz(
        "2188824287183927522224640574525727508854836440041603434369820418657580"
        "8495617",
        10, &modulus);
    Modulus() = Fr(modulus, true);
#endif
  });
}

// static
Fr& FrConfig::Modulus() {
  static base::NoDestructor<Fr> modulus;
  return *modulus;
}

}  // namespace bn254
}  // namespace math
}  // namespace tachyon
