#include "vendors/plonky3/include/baby_bear_row_major_matrix.h"

namespace tachyon::plonky3_api::baby_bear {

RowMajorMatrix::RowMajorMatrix(TachyonBabyBear* ptr, size_t rows, size_t cols)
    : matrix_(tachyon_baby_bear_row_major_matrix_create(
          reinterpret_cast<tachyon_baby_bear*>(ptr), rows, cols)) {}

RowMajorMatrix::~RowMajorMatrix() {
  tachyon_baby_bear_row_major_matrix_destroy(matrix_);
}

size_t RowMajorMatrix::get_rows() const {
  return tachyon_baby_bear_row_major_matrix_get_rows(matrix_);
}

size_t RowMajorMatrix::get_cols() const {
  return tachyon_baby_bear_row_major_matrix_get_cols(matrix_);
}

const TachyonBabyBear* RowMajorMatrix::get_const_data_ptr() const {
  return reinterpret_cast<const TachyonBabyBear*>(
      tachyon_baby_bear_row_major_matrix_get_const_data_ptr(matrix_));
}

std::unique_ptr<RowMajorMatrix> RowMajorMatrix::clone() const {
  return std::make_unique<RowMajorMatrix>(
      tachyon_baby_bear_row_major_matrix_clone(matrix_));
}

std::unique_ptr<RowMajorMatrix> new_row_major_matrix(TachyonBabyBear* ptr,
                                                     size_t rows, size_t cols) {
  return std::make_unique<RowMajorMatrix>(ptr, rows, cols);
}

}  // namespace tachyon::plonky3_api::baby_bear
