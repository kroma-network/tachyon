#ifndef TACHYON_MATH_MATRIX_MATRIX_TO_STRING_OPERATOR_H_
#define TACHYON_MATH_MATRIX_MATRIX_TO_STRING_OPERATOR_H_

#include <sstream>

namespace tachyon::math {

template <typename Matrix>
class MatrixToStringOperator {
 public:
  std::string ToString() const {
    const Matrix* matrix = static_cast<const Matrix*>(this);
    size_t data_idx = 0;
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < matrix->rows(); ++i) {
      if (i != 0) ss << " ";
      ss << "[";
      for (size_t j = 0; j < matrix->cols(); ++j) {
        ss << matrix->data()[data_idx + j];
        if (j != matrix->cols() - 1) {
          ss << ", ";
        }
      }
      ss << "]";
      if (i != matrix->rows() - 1) ss << ",\n";
      data_idx += matrix->stride();
    }
    ss << "]";
    return ss.str();
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_MATRIX_MATRIX_TO_STRING_OPERATOR_H_