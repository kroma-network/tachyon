#include "tachyon/c/math/matrix/baby_bear_row_major_matrix.h"

#include "tachyon/c/math/finite_fields/baby_bear/baby_bear_type_traits.h"
#include "tachyon/c/math/matrix/baby_bear_row_major_matrix_type_traits.h"

using namespace tachyon;

using RowMajorMatrix = math::RowMajorMatrix<math::BabyBear>;

tachyon_baby_bear_row_major_matrix* tachyon_baby_bear_row_major_matrix_create(
    tachyon_baby_bear* ptr, size_t rows, size_t cols) {
  RowMajorMatrix* matrix = new RowMajorMatrix(
      Eigen::Map<RowMajorMatrix>(c::base::native_cast(ptr), rows, cols));
  return c::base::c_cast(matrix);
}

tachyon_baby_bear_row_major_matrix* tachyon_baby_bear_row_major_matrix_clone(
    const tachyon_baby_bear_row_major_matrix* matrix) {
  RowMajorMatrix* cloned = new RowMajorMatrix(c::base::native_cast(*matrix));
  return c::base::c_cast(cloned);
}

void tachyon_baby_bear_row_major_matrix_destroy(
    tachyon_baby_bear_row_major_matrix* matrix) {
  delete c::base::native_cast(matrix);
}

size_t tachyon_baby_bear_row_major_matrix_get_rows(
    const tachyon_baby_bear_row_major_matrix* matrix) {
  return static_cast<size_t>(c::base::native_cast(*matrix).rows());
}

size_t tachyon_baby_bear_row_major_matrix_get_cols(
    const tachyon_baby_bear_row_major_matrix* matrix) {
  return static_cast<size_t>(c::base::native_cast(*matrix).cols());
}

tachyon_baby_bear tachyon_baby_bear_row_major_matrix_get_element(
    const tachyon_baby_bear_row_major_matrix* matrix, size_t row, size_t col) {
  return c::base::c_cast(c::base::native_cast(*matrix)(row, col));
}

const tachyon_baby_bear* tachyon_baby_bear_row_major_matrix_get_const_data_ptr(
    const tachyon_baby_bear_row_major_matrix* matrix) {
  return c::base::c_cast(c::base::native_cast(*matrix).data());
}

tachyon_baby_bear* tachyon_baby_bear_row_major_matrix_get_data_ptr(
    tachyon_baby_bear_row_major_matrix* matrix) {
  return c::base::c_cast(c::base::native_cast(*matrix).data());
}
