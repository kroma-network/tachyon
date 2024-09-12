
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_lde_vec.h"

#include <vector>

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_lde_vec_type_traits.h"

using namespace tachyon;

using LDEVec =
    std::vector<Eigen::Map<const math::RowMajorMatrix<math::BabyBear>>>;

tachyon_sp1_baby_bear_poseidon2_lde_vec*
tachyon_sp1_baby_bear_poseidon2_lde_vec_create() {
  return c::base::c_cast(new LDEVec());
}

void tachyon_sp1_baby_bear_poseidon2_lde_vec_destroy(
    tachyon_sp1_baby_bear_poseidon2_lde_vec* lde_vec) {
  delete c::base::native_cast(lde_vec);
}

void tachyon_sp1_baby_bear_poseidon2_lde_vec_add(
    tachyon_sp1_baby_bear_poseidon2_lde_vec* lde_vec,
    const tachyon_baby_bear* lde, size_t rows, size_t cols) {
  c::base::native_cast(*lde_vec).push_back(
      Eigen::Map<const math::RowMajorMatrix<math::BabyBear>>(
          c::base::native_cast(lde), rows, cols));
}
