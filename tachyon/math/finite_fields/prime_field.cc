#include "tachyon/math/finite_fields/prime_field.h"

#include "absl/base/call_once.h"

#include "tachyon/math/base/gmp_util.h"

namespace tachyon {
namespace math {

// static
void GF7::Init() {
  static absl::once_flag once;
  absl::call_once(once, []() {
#if defined(TACHYON_GMP_BACKEND)
    gmp::UnsignedIntegerToMpz(7, &RawModulus());
#endif
  });
}

}  // namespace math
}  // namespace tachyon
