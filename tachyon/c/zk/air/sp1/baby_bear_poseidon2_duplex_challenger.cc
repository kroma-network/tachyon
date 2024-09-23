#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_duplex_challenger.h"

#include <utility>

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_duplex_challenger_type_traits.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/param_traits/poseidon2_baby_bear.h"
#include "tachyon/crypto/hashes/sponge/poseidon2/poseidon2_params.h"

using namespace tachyon;

using F = math::BabyBear;
using Params =
    crypto::Poseidon2Params<F, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH - 1,
                            TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_ALPHA>;
using Poseidon2 = crypto::Poseidon2Sponge<
    crypto::Poseidon2ExternalMatrix<crypto::Poseidon2Plonky3ExternalMatrix<F>>,
    Params>;

tachyon_sp1_baby_bear_poseidon2_duplex_challenger*
tachyon_sp1_baby_bear_poseidon2_duplex_challenger_create() {
  math::Matrix<F> ark(TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_FULL_ROUNDS +
                          TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
                      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH);
  for (Eigen::Index r = 0; r < ark.rows(); ++r) {
    for (Eigen::Index c = 0; c < ark.cols(); ++c) {
      ark(r, c) = F(kRoundConstants[r][c] % F::Config::kModulus);
    }
  }

  auto config = crypto::Poseidon2Config<Params>::Create(
      crypto::GetPoseidon2InternalShiftArray<Params>(), std::move(ark));
  Poseidon2 sponge(std::move(config));
  return c::base::c_cast(
      new crypto::DuplexChallenger<Poseidon2,
                                   TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE>(
          std::move(sponge)));
}

tachyon_sp1_baby_bear_poseidon2_duplex_challenger*
tachyon_sp1_baby_bear_poseidon2_duplex_challenger_clone(
    const tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger) {
  return c::base::c_cast(
      new crypto::DuplexChallenger<Poseidon2,
                                   TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE>(
          *c::base::native_cast(challenger)));
}

void tachyon_sp1_baby_bear_poseidon2_duplex_challenger_destroy(
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger) {
  delete c::base::native_cast(challenger);
}

void tachyon_sp1_baby_bear_poseidon2_duplex_challenger_observe(
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger,
    const tachyon_baby_bear* value) {
  c::base::native_cast(challenger)->Observe(c::base::native_cast(*value));
}

tachyon_baby_bear tachyon_sp1_baby_bear_poseidon2_duplex_challenger_sample(
    tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger) {
  return c::base::c_cast(c::base::native_cast(challenger)->Sample());
}
