#include "tachyon/c/zk/base/bn254_blinder.h"

#include "tachyon/c/zk/base/bn254_blinder_type_traits.h"

using namespace tachyon;

void tachyon_halo2_bn254_blinder_set_blinding_factors(
    tachyon_bn254_blinder* blinder, uint32_t blinding_factors) {
  c::base::native_cast(blinder)->set_blinding_factors(blinding_factors);
}

uint32_t tachyon_halo2_bn254_blinder_get_blinding_factors(
    const tachyon_bn254_blinder* blinder) {
  return c::base::native_cast(blinder)->blinding_factors();
}
