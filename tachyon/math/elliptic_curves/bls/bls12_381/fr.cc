#include "tachyon/math/elliptic_curves/bls/bls12_381/fr.h"

#include "absl/base/call_once.h"

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/gmp_util.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

// static
void FrConfig::Init() {
  static absl::once_flag once;
  absl::call_once(once, []() {
#if defined(TACHYON_GMP_BACKEND)
    mpz_class modulus;
    // Hex: 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
    gmp::MustParseIntoMpz(
        "5243587517512619047944774050818596583769055250052763782260365869993858"
        "1184513",
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

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon
