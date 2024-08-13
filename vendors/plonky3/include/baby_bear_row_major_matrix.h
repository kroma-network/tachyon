#ifndef VENDORS_PLONKY3_INCLUDE_BABY_BEAR_ROW_MAJOR_MATRIX_H_
#define VENDORS_PLONKY3_INCLUDE_BABY_BEAR_ROW_MAJOR_MATRIX_H_

#include <memory>

#include "rust/cxx.h"

#include "tachyon/c/math/matrix/baby_bear_row_major_matrix.h"

namespace tachyon::plonky3_api::baby_bear {

struct TachyonBabyBear;

class RowMajorMatrix {
 public:
  RowMajorMatrix(TachyonBabyBear* ptr, size_t rows, size_t cols);
  explicit RowMajorMatrix(tachyon_baby_bear_row_major_matrix* matrix)
      : matrix_(matrix) {}
  RowMajorMatrix(const RowMajorMatrix& other) = delete;
  RowMajorMatrix& operator=(const RowMajorMatrix& other) = delete;
  ~RowMajorMatrix();

  size_t get_rows() const;
  size_t get_cols() const;
  const TachyonBabyBear* get_const_data_ptr() const;
  std::unique_ptr<RowMajorMatrix> clone() const;

 private:
  tachyon_baby_bear_row_major_matrix* matrix_;
};

std::unique_ptr<RowMajorMatrix> new_row_major_matrix(TachyonBabyBear* ptr,
                                                     size_t rows, size_t cols);

}  // namespace tachyon::plonky3_api::baby_bear

#endif  // VENDORS_PLONKY3_INCLUDE_BABY_BEAR_ROW_MAJOR_MATRIX_H_
