#include "tachyon/c/zk/base/bn254_blinder.h"

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/zk/base/blinder.h"

using namespace tachyon;

void tachyon_halo2_bn254_blinder_set_blinding_factors(
    tachyon_bn254_blinder* blinder, uint32_t blinding_factors) {
  reinterpret_cast<zk::Blinder<math::bn254::Fr>*>(blinder)
      ->set_blinding_factors(blinding_factors);
}

uint32_t tachyon_halo2_bn254_blinder_get_blinding_factors(
    const tachyon_bn254_blinder* blinder) {
  return reinterpret_cast<const zk::Blinder<math::bn254::Fr>*>(blinder)
      ->blinding_factors();
}
