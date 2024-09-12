#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_commitment_vec.h"

#include <array>
#include <vector>

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_commitment_vec_type_traits.h"

using namespace tachyon;

using CommitmentVec = std::vector<
    std::array<math::BabyBear, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK>>;

tachyon_sp1_baby_bear_poseidon2_commitment_vec*
tachyon_sp1_baby_bear_poseidon2_commitment_vec_create(size_t rounds) {
  return c::base::c_cast(new CommitmentVec(rounds));
}

void tachyon_sp1_baby_bear_poseidon2_commitment_vec_destroy(
    tachyon_sp1_baby_bear_poseidon2_commitment_vec* commitment_vec) {
  delete c::base::native_cast(commitment_vec);
}

void tachyon_sp1_baby_bear_poseidon2_commitment_vec_set(
    tachyon_sp1_baby_bear_poseidon2_commitment_vec* commitment_vec,
    size_t round, const tachyon_baby_bear* commitment) {
  std::array<math::BabyBear, TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK>&
      native_commitment = c::base::native_cast(*commitment_vec)[round];
  for (size_t i = 0; i < TACHYON_PLONKY3_BABY_BEAR_POSEIDON2_CHUNK; ++i) {
    native_commitment[i] = c::base::native_cast(commitment[i]);
  }
}
