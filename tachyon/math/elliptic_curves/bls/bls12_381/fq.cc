#include "tachyon/math/elliptic_curves/bls/bls12_381/fq.h"

#include "absl/base/call_once.h"

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/gmp/gmp_util.h"

namespace tachyon {
namespace math {
namespace bls12_381 {

// static
void FqConfig::Init() {
  static absl::once_flag once;
  absl::call_once(once, []() {
#if defined(TACHYON_GMP_BACKEND)
    mpz_class modulus;
    // Hex:
    // 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
    gmp::MustParseIntoMpz(
        "4002409555221667393417789825735904156556882819939007885332058136124031"
        "650490837864442687629129015664037894272559787",
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

}  // namespace bls12_381
}  // namespace math
}  // namespace tachyon
