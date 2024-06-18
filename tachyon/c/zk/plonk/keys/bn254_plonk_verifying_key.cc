#include "tachyon/c/zk/plonk/keys/bn254_plonk_verifying_key.h"

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/zk/plonk/keys/verifying_key.h"

using namespace tachyon;

using VKey =
    zk::plonk::VerifyingKey<math::bn254::Fr, math::bn254::G1AffinePoint>;

const tachyon_bn254_plonk_constraint_system*
tachyon_bn254_plonk_verifying_key_get_constraint_system(
    const tachyon_bn254_plonk_verifying_key* vk) {
  const VKey* cpp_vk = reinterpret_cast<const VKey*>(vk);
  return reinterpret_cast<const tachyon_bn254_plonk_constraint_system*>(
      &cpp_vk->constraint_system());
}

tachyon_bn254_fr tachyon_bn254_plonk_verifying_key_get_transcript_repr(
    const tachyon_bn254_plonk_verifying_key* vk) {
  return c::base::c_cast(reinterpret_cast<const VKey*>(vk)->transcript_repr());
}
