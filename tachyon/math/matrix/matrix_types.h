#ifndef TACHYON_MATH_MATRIX_MATRIX_TYPES_H_
#define TACHYON_MATH_MATRIX_MATRIX_TYPES_H_

#include <type_traits>
#include <utility>

#include "Eigen/Core"

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/math/finite_fields/prime_field_base.h"

namespace tachyon {
namespace math {

template <typename Field, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic,
          int Options = 0, int MaxRows = Rows, int MaxCols = Cols>
using Matrix = Eigen::Matrix<Field, Rows, Cols, Options, MaxRows, MaxCols>;

template <typename Field, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic,
          int MaxRows = Rows, int MaxCols = Cols>
using ColMajorMatrix =
    Eigen::Matrix<Field, Rows, Cols, Eigen::ColMajor, MaxRows, MaxCols>;

template <typename Field, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic,
          int MaxRows = Rows, int MaxCols = Cols>
using RowMajorMatrix =
    Eigen::Matrix<Field, Rows, Cols, Eigen::RowMajor, MaxRows, MaxCols>;

template <typename Field, int Size = Eigen::Dynamic, int MaxSize = Size>
using DiagonalMatrix = Eigen::DiagonalMatrix<Field, Size, MaxSize>;

template <typename Field, int Rows = Eigen::Dynamic, int Options = 0,
          int MaxRows = Rows>
using Vector = Eigen::Matrix<Field, Rows, 1, Options, MaxRows, 1>;

template <typename Field, int Cols = Eigen::Dynamic, int Options = 0,
          int MaxCols = Cols>
using RowVector = Eigen::Matrix<Field, 1, Cols, Options, 1, MaxCols>;

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

template <typename Field, int Size, int MaxSize>
class Copyable<Eigen::DiagonalMatrix<Field, Size, MaxSize>> {
 public:
  using DiagonalMatrix = Eigen::DiagonalMatrix<Field, Size, MaxSize>;
  using DiagonalVector = typename DiagonalMatrix::DiagonalVectorType;

  static bool WriteTo(const DiagonalMatrix& matrix, Buffer* buffer) {
    if (!buffer->WriteMany(matrix.rows())) return false;
    const DiagonalVector& diagonal = matrix.diagonal();
    for (Eigen::Index i = 0; i < diagonal.size(); ++i) {
      if (!buffer->Write(diagonal.data()[i])) return false;
    }
    return true;
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, DiagonalMatrix* matrix) {
    Eigen::Index size;
    DiagonalVector vector_tmp;
    if (!buffer.ReadMany(&size)) return false;
    if (Size != Eigen::Dynamic) {
      if (size != Size) return false;
    } else {
      vector_tmp.resize(size);
    }
    for (Eigen::Index i = 0; i < vector_tmp.size(); ++i) {
      if (!buffer.Read(&vector_tmp.data()[i])) return false;
    }
    *matrix = DiagonalMatrix(std::move(vector_tmp));
    return true;
  }

  static size_t EstimateSize(const DiagonalMatrix& matrix) {
    return matrix.rows() * sizeof(Field) + sizeof(Eigen::Index);
  }
};

}  // namespace base
}  // namespace tachyon

namespace Eigen::internal {

template <typename T>
struct scalar_random_op<
    T,
    std::enable_if_t<std::is_base_of_v<tachyon::math::PrimeFieldBase<T>, T>>> {
  inline const T operator()() const { return T::Random(); }
};

}  // namespace Eigen::internal

#endif  // TACHYON_MATH_MATRIX_MATRIX_TYPES_H_
