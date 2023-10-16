// Copyright cuZK authors.
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.cuzk and the LICENCE-APACHE.cuzk
// file.

#ifndef TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_CUZK_CUZK_CSR_SPARSE_MATRIX_H_
#define TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_CUZK_CUZK_CSR_SPARSE_MATRIX_H_

#include <string>

#include "tachyon/export.h"

namespace tachyon::math {

// NOTE: Make sure this is copyable!
struct TACHYON_EXPORT CUZKCSRSparseMatrix {
  struct Element {
    unsigned int index = 0;
    unsigned int data_addr = 0;

    std::string ToString() const;
  };

  std::string ToString() const;

  // not owned
  unsigned int* row_ptrs = nullptr;
  // not owned
  Element* col_datas = nullptr;
  unsigned int col_datas_size = 0;
  unsigned int rows = 0;
  unsigned int cols = 0;
};
}  // namespace tachyon::math

#endif  // TACHYON_MATH_ELLIPTIC_CURVES_MSM_ALGORITHMS_CUZK_CUZK_CSR_SPARSE_MATRIX_H_
