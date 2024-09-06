#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opened_values.h"

#include <vector>

#include "tachyon/base/auto_reset.h"
#include "tachyon/base/buffer/buffer.h"
#include "tachyon/c/math/finite_fields/baby_bear/baby_bear4_type_traits.h"
#include "tachyon/c/zk/air/sp1/baby_bear_poseidon2_opened_values_type_traits.h"

using namespace tachyon;

using OpenedValues =
    std::vector<std::vector<std::vector<std::vector<math::BabyBear4>>>>;

tachyon_sp1_baby_bear_poseidon2_opened_values*
tachyon_sp1_baby_bear_poseidon2_opened_values_create(size_t rounds) {
  return c::base::c_cast(new OpenedValues(rounds));
}

tachyon_sp1_baby_bear_poseidon2_opened_values*
tachyon_sp1_baby_bear_poseidon2_opened_values_clone(
    const tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values) {
  return c::base::c_cast(
      new OpenedValues(c::base::native_cast(*opened_values)));
}

void tachyon_sp1_baby_bear_poseidon2_opened_values_destroy(
    tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values) {
  delete c::base::native_cast(opened_values);
}

void tachyon_sp1_baby_bear_poseidon2_opened_values_serialize(
    const tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values,
    uint8_t* data, size_t* data_len) {
  if (data == nullptr) {
    *data_len = base::EstimateSize(c::base::native_cast(*opened_values));
    return;
  }
  base::AutoReset<bool> auto_reset(
      &base::Copyable<math::BabyBear>::s_is_in_montgomery, true);
  base::Buffer buffer(data, *data_len);
  CHECK(buffer.Write(c::base::native_cast(*opened_values)));
}

void tachyon_sp1_baby_bear_poseidon2_opened_values_allocate_outer(
    tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values, size_t round,
    size_t rows, size_t cols) {
  std::vector<std::vector<std::vector<math::BabyBear4>>>& mat =
      c::base::native_cast(*opened_values)[round];
  mat.resize(rows);
  for (size_t r = 0; r < rows; ++r) {
    mat[r].resize(cols);
  }
}

void tachyon_sp1_baby_bear_poseidon2_opened_values_allocate_inner(
    tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values, size_t round,
    size_t r, size_t cols, size_t size) {
  std::vector<std::vector<std::vector<math::BabyBear4>>>& mat =
      c::base::native_cast(*opened_values)[round];
  std::vector<std::vector<math::BabyBear4>>& row = mat[r];
  for (size_t c = 0; c < cols; ++c) {
    row[c].resize(size);
  }
}

void tachyon_sp1_baby_bear_poseidon2_opened_values_set(
    tachyon_sp1_baby_bear_poseidon2_opened_values* opened_values, size_t round,
    size_t row, size_t col, size_t idx, const tachyon_baby_bear4* value) {
  c::base::native_cast(*opened_values)[round][row][col][idx] =
      c::base::native_cast(*value);
}
