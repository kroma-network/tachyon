#include "tachyon/c/zk/plonk/halo2/bn254_instance_columns_vec.h"

#include <vector>

#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

using namespace tachyon;

using ColumnsVec = std::vector<std::vector<std::vector<math::bn254::Fr>>>;

tachyon_halo2_bn254_instance_columns_vec*
tachyon_halo2_bn254_instance_columns_vec_create(size_t num_circuits) {
  return reinterpret_cast<tachyon_halo2_bn254_instance_columns_vec*>(
      new ColumnsVec(num_circuits));
}

void tachyon_halo2_bn254_instance_columns_vec_destroy(
    tachyon_halo2_bn254_instance_columns_vec* data) {
  delete reinterpret_cast<ColumnsVec*>(data);
}

void tachyon_halo2_bn254_instance_columns_vec_resize_columns(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t num_columns) {
  reinterpret_cast<ColumnsVec&>(*data)[circuit_idx].resize(num_columns);
}

void tachyon_halo2_bn254_instance_columns_vec_reserve_values(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t column_idx, size_t num_values) {
  reinterpret_cast<ColumnsVec&>(*data)[circuit_idx][column_idx].reserve(
      num_values);
}

void tachyon_halo2_bn254_instance_columns_vec_add_values(
    tachyon_halo2_bn254_instance_columns_vec* data, size_t circuit_idx,
    size_t column_idx, const tachyon_bn254_fr* value) {
  reinterpret_cast<ColumnsVec&>(*data)[circuit_idx][column_idx].push_back(
      reinterpret_cast<const math::bn254::Fr&>(*value));
}
