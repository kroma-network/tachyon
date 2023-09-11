#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_CUZK_ELL_SPARSE_MATRIX_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_CUZK_ELL_SPARSE_MATRIX_H_

#include <string>

#include "tachyon/export.h"

namespace tachyon::math {

// NOTE: Make sure this is copyable!
struct TACHYON_EXPORT CUZKELLSparseMatrix {
  constexpr void Insert(unsigned int row, unsigned int col) {
    size_t idx = row * cols + row_lengths[row];
    ++row_lengths[row];
    col_indices[idx] = col;
  }

  std::string ToString() const;

  // not owned
  unsigned int* row_lengths = nullptr;
  // not owned
  unsigned int* col_indices = nullptr;
  unsigned int rows = 0;
  unsigned int cols = 0;
};
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_CUZK_ELL_SPARSE_MATRIX_H_
