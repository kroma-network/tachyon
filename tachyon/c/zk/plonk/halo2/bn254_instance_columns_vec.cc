#include "tachyon/c/zk/plonk/halo2/bn254_instance_columns_vec.h"

#include <vector>

#include "tachyon/c/math/elliptic_curves/bn/bn254/fr_type_traits.h"
#include "tachyon/c/zk/plonk/halo2/bn254_instance_columns_vec_type_traits.h"

using namespace tachyon;

using ColumnsVec = std::vector<std::vector<std::vector<math::bn254::Fr>>>;

tachyon_halo2_bn254_instance_columns_vec*
tachyon_halo2_bn254_instance_columns_vec_create(size_t num_circuits) {
  return c::base::c_cast(new ColumnsVec(num_circuits));
}

void tachyon_halo2_bn254_instance_columns_vec_destroy(
    tachyon_halo2_bn254_instance_columns_vec* data) {
  delete c::base::native_cast(data);
}

void tachyon_halo2_bn254_instance_columns_vec_resize_columns(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t num_columns) {
  c::base::native_cast(*data)[circuit_idx].resize(num_columns);
}

void tachyon_halo2_bn254_instance_columns_vec_reserve_values(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t column_idx, size_t num_values) {
  c::base::native_cast(*data)[circuit_idx][column_idx].reserve(num_values);
}

void tachyon_halo2_bn254_instance_columns_vec_add_values(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t column_idx, const tachyon_bn254_fr* value) {
  c::base::native_cast(*data)[circuit_idx][column_idx].push_back(
      c::base::native_cast(*value));
}
