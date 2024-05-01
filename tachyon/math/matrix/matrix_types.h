#ifndef TACHYON_MATH_MATRIX_MATRIX_TYPES_H_
#define TACHYON_MATH_MATRIX_MATRIX_TYPES_H_

#include <utility>

#include "Eigen/Core"

#include "tachyon/base/buffer/copyable.h"

namespace tachyon {
namespace math {

template <typename Field>
using Matrix = Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Field>
using Vector = Eigen::Matrix<Field, Eigen::Dynamic, 1>;

template <typename Field>
using RowVector = Eigen::Matrix<Field, 1, Eigen::Dynamic>;

}  // namespace math

namespace base {

template <typename Field, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
class Copyable<Eigen::Matrix<Field, Rows, Cols, Options, MaxRows, MaxCols>> {
 public:
  using Matrix = Eigen::Matrix<Field, Rows, Cols, Options, MaxRows, MaxCols>;

  static bool WriteTo(const Matrix& matrix, Buffer* buffer) {
    if (!buffer->WriteMany(matrix.rows(), matrix.cols())) return false;
    for (Eigen::Index i = 0; i < matrix.size(); ++i) {
      if (!buffer->Write(matrix.data()[i])) return false;
    }
    return true;
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, Matrix* matrix) {
    Eigen::Index rows, cols;
    Matrix matrix_tmp;
    if (!buffer.ReadMany(&rows, &cols)) return false;
    if (Rows != Eigen::Dynamic && Cols != Eigen::Dynamic) {
      if (rows != Rows || cols != Cols) return false;
    } else if (Rows != Eigen::Dynamic) {
      if (rows != Rows) return false;
      matrix_tmp.resize(Eigen::NoChange, cols);
    } else if (Cols != Eigen::Dynamic) {
      if (cols != Cols) return false;
      matrix_tmp.resize(rows, Eigen::NoChange);
    } else {
      matrix_tmp.resize(rows, cols);
    }
    for (Eigen::Index i = 0; i < matrix_tmp.size(); ++i) {
      if (!buffer.Read(&matrix_tmp.data()[i])) return false;
    }
    *matrix = std::move(matrix_tmp);
    return true;
  }

  static size_t EstimateSize(const Matrix& matrix) {
    return matrix.size() * sizeof(Field) + sizeof(Eigen::Index) * 2;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_MATH_MATRIX_MATRIX_TYPES_H_
