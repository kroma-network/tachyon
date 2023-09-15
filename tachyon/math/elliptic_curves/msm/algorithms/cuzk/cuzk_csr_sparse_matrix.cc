#include "tachyon/math/elliptic_curves/msm/algorithms/cuzk/cuzk_csr_sparse_matrix.h"

#include <sstream>

#include "absl/strings/substitute.h"

namespace tachyon::math {

std::string CUZKCSRSparseMatrix::Element::ToString() const {
  return absl::Substitute("($0, $1)", index, data_addr);
}

std::string CUZKCSRSparseMatrix::ToString() const {
  std::stringstream ss;
  ss << "===ROW PTRS===" << std::endl;
  for (unsigned int i = 0; i < rows + 1; ++i) {
    ss << row_ptrs[i] << std::endl;
  }

  ss << "===COL DATAS===" << std::endl;
  for (unsigned int i = 0; i < col_datas_size; ++i) {
    ss << col_datas[i].ToString() << std::endl;
  }
  return ss.str();
}

}  // namespace tachyon::math
