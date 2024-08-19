#include "tachyon/c/zk/air/plonky3/challenger/baby_bear_poseidon2_duplex_challenger.h"

#include <utility>

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/zk/air/plonky3/challenger/baby_bear_poseidon2_duplex_challenger_type_traits.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"

using namespace tachyon;

using F = math::BabyBear;
using Poseidon2 = crypto::Poseidon2Sponge<
    crypto::Poseidon2ExternalMatrix<crypto::Poseidon2Plonky3ExternalMatrix<F>>>;

tachyon_plonky3_baby_bear_poseidon2_duplex_challenger*
tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_create() {
  crypto::Poseidon2Config<F> config = crypto::Poseidon2Config<F>::CreateCustom(
      15, 7, 8, 13, math::GetPoseidon2BabyBearInternalShiftVector<15>());
  Poseidon2 sponge(std::move(config));
  return c::base::c_cast(
      new zk::air::plonky3::DuplexChallenger<Poseidon2, 16, 8>(
          std::move(sponge)));
}

tachyon_plonky3_baby_bear_poseidon2_duplex_challenger*
tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_clone(
    const tachyon_plonky3_baby_bear_poseidon2_duplex_challenger* challenger) {
  zk::air::plonky3::DuplexChallenger<Poseidon2, 16, 8>* cloned_challenger =
      new zk::air::plonky3::DuplexChallenger<Poseidon2, 16, 8>(
          *c::base::native_cast(challenger));
  return c::base::c_cast(cloned_challenger);
}

void tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_destroy(
    tachyon_plonky3_baby_bear_poseidon2_duplex_challenger* challenger) {
  delete c::base::native_cast(challenger);
}

void tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_observe(
    tachyon_plonky3_baby_bear_poseidon2_duplex_challenger* challenger,
    const tachyon_baby_bear* value) {
  c::base::native_cast(challenger)->Observe(c::base::native_cast(*value));
}

tachyon_baby_bear tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_sample(
    tachyon_plonky3_baby_bear_poseidon2_duplex_challenger* challenger) {
  return c::base::c_cast(c::base::native_cast(challenger)->Sample());
}
