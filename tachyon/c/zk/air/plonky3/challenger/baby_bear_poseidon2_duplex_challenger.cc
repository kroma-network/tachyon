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
      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH - 1,
      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_ALPHA,
      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_FULL_ROUNDS,
      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
      math::GetPoseidon2BabyBearInternalShiftVector<
          TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH - 1>());
  Poseidon2 sponge(std::move(config));
  return c::base::c_cast(
      new zk::air::plonky3::DuplexChallenger<
          Poseidon2, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH,
          TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE>(std::move(sponge)));
}

tachyon_plonky3_baby_bear_poseidon2_duplex_challenger*
tachyon_plonky3_baby_bear_poseidon2_duplex_challenger_clone(
    const tachyon_plonky3_baby_bear_poseidon2_duplex_challenger* challenger) {
  return c::base::c_cast(new zk::air::plonky3::DuplexChallenger<
                         Poseidon2, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH,
                         TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE>(
      *c::base::native_cast(challenger)));
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
