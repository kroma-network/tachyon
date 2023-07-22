#ifndef TACHYON_MATH_MATRIX_MATRIX_VIEW_H_
#define TACHYON_MATH_MATRIX_MATRIX_VIEW_H_

#include <utility>

#include "tachyon/base/logging.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/matrix/matrix_to_string_operator.h"
#include "tachyon/math/matrix/matrix_traits.h"

namespace tachyon {
namespace math {

template <typename T, size_t Rows_, size_t Cols_>
class Matrix;

// For the given 3x3 matrix,
//
//   1  2  3
//   4  5  6
//   7  8  9
//
// To access 5, we compute the index as 1 * |Rows(3)| + 1.
// However, accessing a submatrix is different. For the same matrix,
// if we take a MatrixView using |Block(1, 1)|, it gives us:
//
//   5  6
//   8  9
//
// Here, to access 9, we shouldn't simply compute the index as 1 * |Rows(2)|
// + 1. In this example, we need to consider the |stride|. The |stride| is
// always the same as the number of columns in the original matrix. Since the
// original matrix was 3x3, we can access 9 by indexing as 1 * |Stride(3)| + 1.
template <typename T>
class MatrixView : public MatrixToStringOperator<MatrixView<T>> {
 public:
  typedef T value_type;

  constexpr MatrixView(const T* data, size_t rows, size_t cols, size_t stride)
      : data_(data), rows_(rows), cols_(cols), stride_(stride) {}
  MatrixView(const MatrixView& other) = delete;
  MatrixView& operator=(const MatrixView& other) = delete;
  constexpr MatrixView(MatrixView&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        rows_(other.rows_),
        cols_(other.cols_),
        stride_(other.stride_) {}
  constexpr MatrixView& operator=(MatrixView&& other) {
    data_ = std::exchange(other.data_, nullptr);
    rows_ = other.rows_;
    cols_ = other.cols_;
    stride_ = other.stride_;
    return *this;
  }

  constexpr const T* data() const { return data_; }

  constexpr size_t rows() const { return rows_; }
  constexpr size_t cols() const { return cols_; }
  constexpr size_t stride() const { return stride_; }

  constexpr const T& at(size_t row, size_t col) const {
    DCHECK_LT(row, rows_);
    DCHECK_LT(col, cols_);
    return data_[row * stride_ + col];
  }

  constexpr bool IsZero() const {
    size_t data_idx = 0;
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        if (!math::IsZero(data_[data_idx + j])) return false;
      }
      data_idx += stride_;
    }
    return true;
  }

  constexpr bool IsIdentity() const {
    if (!IsSquare()) return false;
    size_t data_idx = 0;
    for (size_t i = 0; i < rows_; ++i) {
      if (!math::IsOne(data_[data_idx])) return false;
      data_idx += stride_ + 1;
    }
    return true;
  }

  constexpr bool IsSquare() const { return rows_ == cols_; }

  constexpr T Trace() const {
    CHECK(IsSquare()) << "Trace is only defined for square matrix";
    T sum = Zero<T>();
    size_t data_idx = 0;
    for (size_t i = 0; i < rows_; ++i) {
      sum += data_[data_idx];
      data_idx += stride_ + 1;
    }
    return sum;
  }

  MatrixView Block(size_t row, size_t col, size_t rows = 0,
                   size_t cols = 0) const {
    DCHECK_LT(row, rows_);
    DCHECK_LT(col, cols_);
    if (rows == 0) {
      rows = rows_ - row;
    } else {
      DCHECK_LE(rows, rows_ - row);
    }
    if (cols == 0) {
      cols = cols_ - col;
    } else {
      DCHECK_LE(cols, cols_ - col);
    }
    return {&data_[row * stride_ + col], rows, cols, stride_};
  }

  constexpr bool operator==(const MatrixView& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) return false;
    size_t data_idx = 0;
    size_t data_idx2 = 0;
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        if (data_[data_idx + j] != other.data_[data_idx2 + j]) return false;
      }
      data_idx += stride_;
      data_idx2 += other.stride_;
    }
    return true;
  }

  constexpr bool operator!=(const MatrixView& other) const {
    return !operator==(other);
  }

  template <size_t Rows, size_t Cols>
  constexpr bool operator==(const Matrix<T, Rows, Cols>& other) const {
    if (rows_ != Rows || cols_ != Cols) return false;
    size_t data_idx = 0;
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        if (data_[data_idx + j] != other.at(i, j)) return false;
      }
      data_idx += stride_;
    }
    return true;
  }

  template <size_t Rows, size_t Cols>
  constexpr bool operator!=(const Matrix<T, Rows, Cols>& other) const {
    return !operator==(other);
  }

 private:
  const T* data_;
  const size_t rows_;
  const size_t cols_;
  const size_t stride_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const MatrixView<T>& matrix) {
  return os << matrix.ToString();
}

template <typename T>
struct MatrixTraits<MatrixView<T>> {
 public:
  constexpr static size_t Rows = kDynamic;
  constexpr static size_t Cols = kDynamic;
  constexpr static size_t Size = kDynamic;

  constexpr static bool is_view = true;

  typedef T value_type;
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_MATRIX_MATRIX_VIEW_H_