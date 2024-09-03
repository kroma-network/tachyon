#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opening_points.h"

#include <vector>

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear4_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opening_points_type_traits.h"

using namespace tachyon;

using OpeningPoints = std::vector<std::vector<std::vector<math::BabyBear4>>>;

tachyon_sp1_baby_bear_poseidon2_opening_points*
tachyon_sp1_baby_bear_poseidon2_opening_points_create(size_t rounds) {
  return c::base::c_cast(new OpeningPoints(rounds));
}

tachyon_sp1_baby_bear_poseidon2_opening_points*
tachyon_sp1_baby_bear_poseidon2_opening_points_clone(
    const tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points) {
  return c::base::c_cast(
      new OpeningPoints(c::base::native_cast(*opening_points)));
}

void tachyon_sp1_baby_bear_poseidon2_opening_points_destroy(
    tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points) {
  delete c::base::native_cast(opening_points);
}

void tachyon_sp1_baby_bear_poseidon2_opening_points_allocate(
    tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points,
    size_t round, size_t rows, size_t cols) {
  std::vector<std::vector<math::BabyBear4>>& mat =
      c::base::native_cast(*opening_points)[round];
  mat.resize(rows);
  for (size_t r = 0; r < rows; ++r) {
    mat[r].resize(cols);
  }
}

void tachyon_sp1_baby_bear_poseidon2_opening_points_set(
    tachyon_sp1_baby_bear_poseidon2_opening_points* opening_points,
    size_t round, size_t row, size_t col, const tachyon_baby_bear4* point) {
  c::base::native_cast(*opening_points)[round][row][col] =
      c::base::native_cast(*point);
}
