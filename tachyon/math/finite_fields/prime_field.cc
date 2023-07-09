#include "tachyon/math/finite_fields/prime_field.h"

#include "absl/base/call_once.h"

#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/gmp_util.h"

namespace tachyon {
namespace math {

// static
void GF7Config::Init() {
  static absl::once_flag once;
  absl::call_once(once, []() {
#if defined(TACHYON_GMP_BACKEND)
    mpz_class modulus;
    gmp::UnsignedIntegerToMpz(7, &modulus);
    Modulus() = GF7(modulus, true);
#endif
  });
}

// static
GF7& GF7Config::Modulus() {
  static base::NoDestructor<GF7> modulus;
  return *modulus;
}

}  // namespace math
}  // namespace tachyon
