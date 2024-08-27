#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_duplex_challenger.h"

#include <utility>

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_duplex_challenger_type_traits.h"
#include "tachyon/math/finite_fields/baby_bear/poseidon2.h"

using namespace tachyon;

using F = math::BabyBear;
using Poseidon2 = crypto::Poseidon2Sponge<
    crypto::Poseidon2ExternalMatrix<crypto::Poseidon2Plonky3ExternalMatrix<F>>>;

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

  crypto::Poseidon2Config<F> config = crypto::Poseidon2Config<F>::CreateCustom(
      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH - 1,
      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_ALPHA,
      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_FULL_ROUNDS,
      TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_PARTIAL_ROUNDS,
      math::GetPoseidon2BabyBearInternalShiftVector<
          TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH - 1>(),
      std::move(ark));
  Poseidon2 sponge(std::move(config));
  return c::base::c_cast(
      new zk::air::plonky3::DuplexChallenger<
          Poseidon2, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH,
          TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_RATE>(std::move(sponge)));
}

tachyon_sp1_baby_bear_poseidon2_duplex_challenger*
tachyon_sp1_baby_bear_poseidon2_duplex_challenger_clone(
    const tachyon_sp1_baby_bear_poseidon2_duplex_challenger* challenger) {
  return c::base::c_cast(new zk::air::plonky3::DuplexChallenger<
                         Poseidon2, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_WIDTH,
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
