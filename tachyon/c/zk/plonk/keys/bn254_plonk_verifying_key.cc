#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/c/zk/plonk/constraint_system/bn254_constraint_system_type_traits.h"
#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key_type_traits.h"

using namespace tachyon;

const tachyon_bn254_plonk_constraint_system*
tachyon_bn254_plonk_verifying_key_get_constraint_system(
    const tachyon_bn254_plonk_verifying_key* vk) {
  return c::base::c_cast(&c::base::native_cast(vk)->constraint_system());
}

tachyon_bn254_fr tachyon_bn254_plonk_verifying_key_get_transcript_repr(
    const tachyon_bn254_plonk_verifying_key* vk) {
  return c::base::c_cast(c::base::native_cast(vk)->transcript_repr());
}
