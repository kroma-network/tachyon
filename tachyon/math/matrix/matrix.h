#ifndef TACHYON_MATH_MATRIX_MATRIX_H_
#define TACHYON_MATH_MATRIX_MATRIX_H_

#include <string.h>

#include <cmath>
#include <sstream>

#include "tachyon/base/logging.h"
#include "tachyon/base/no_destructor.h"
#include "tachyon/math/base/identities.h"
#include "tachyon/math/base/rings.h"
#include "tachyon/math/matrix/matrix_inverse_operator.h"
#include "tachyon/math/matrix/matrix_to_string_operator.h"
#include "tachyon/math/matrix/matrix_view.h"

namespace tachyon {
namespace math {

// This class is aiming for stack allocated simple 2-D matrix.
// If you want to use complicated matrix operation, please use another 3rd party
// libraries!
template <typename T, size_t Rows_, size_t Cols_>
class Matrix : public Ring<Matrix<T, Rows_, Cols_>>,
               public MatrixToStringOperator<Matrix<T, Rows_, Cols_>>,
               public MatInverseOperator<Matrix<T, Rows_, Cols_>> {
 public:
  static constexpr size_t Rows = Rows_;
  static constexpr size_t Cols = Cols_;
  static constexpr size_t Size = Rows * Cols;

  static_assert(Rows != 0 && Cols != 0, "Invalid rows or cols");

  typedef T value_type;

  constexpr Matrix() {}
  // 1x1 Matrix
  constexpr explicit Matrix(T v0);
  // 1x2 or 2x1 Matrix
  constexpr Matrix(T v0, T v1);
  // 1x3 or 3x1 Matrix
  constexpr Matrix(T v0, T v1, T v2);
  // 1x4, 2x2 or 4x1 Matrix
  constexpr Matrix(T v0, T v1, T v2, T v3);
  // 1x5 or 5x1 Matrix
  constexpr Matrix(T v0, T v1, T v2, T v3, T v4);
  // 1x6, 2x3, 3x2 or 6x1 Matrix
  constexpr Matrix(T v0, T v1, T v2, T v3, T v4, T v5);
  // 1x7 or 7x1 Matrix
  constexpr Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6);
  // 1x8, 2x4, 4x2 or 8x1 Matrix
  constexpr Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7);
  // 1x9, 3x3 or 9x1 Matrix
  constexpr Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8);
  // 1x10, 2x5, 5x2 or 10x1 Matrix
  constexpr Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9);
  // 1x12, 2x6, 3x4, 4x3, 6x2 or 12x1 Matrix
  constexpr Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9,
                   T v10, T v11);
  // 1x16, 2x8, 4x4, 8x2 or 16x1 Matrix
  constexpr Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9,
                   T v10, T v11, T v12, T v13, T v14, T v15);
  constexpr Matrix(const T* data) {
    for (size_t i = 0; i < Size; ++i) {
      data_[i] = data[i];
    }
  }

  constexpr static Matrix Zero() { return {}; }
  constexpr static Matrix Identity() {
    static_assert(Rows_ == Cols_, "Identity is only defined for square matrix");
    Matrix identity;
    for (size_t i = 0; i < Size; i += (Rows + 1)) {
      identity[i] = T::One();
    }
    return identity;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }

  constexpr size_t rows() const { return Rows; }
  constexpr size_t cols() const { return Cols; }
  constexpr size_t stride() const { return Cols; }

  constexpr T& operator[](size_t i) {
    DCHECK_LT(i, Size);
    return data_[i];
  }
  constexpr const T& operator[](size_t i) const {
    DCHECK_LT(i, Size);
    return data_[i];
  }

  constexpr T& at(size_t row, size_t col) {
    DCHECK_LT(row, Rows);
    DCHECK_LT(col, Cols);
    return data_[row * Cols + col];
  }
  constexpr const T& at(size_t row, size_t col) const {
    DCHECK_LT(row, Rows);
    DCHECK_LT(col, Cols);
    return data_[row * Cols + col];
  }

  constexpr bool IsZero() const {
    for (size_t i = 0; i < Size; ++i) {
      if (!data_[i].IsZero()) return false;
    }
    return true;
  }

  constexpr bool IsIdentity() const {
    for (size_t i = 0; i < Size; i += (Rows + 1)) {
      if (!data_[i].IsOne()) return false;
    }
    return true;
  }

  constexpr bool IsSquare() const { return Rows == Cols; }

  constexpr T Trace() const {
    CHECK(IsSquare()) << "Trace is only defined for square matrix";
    T sum = T::Zero();
    for (size_t i = 0; i < Size; i += (Rows + 1)) {
      sum += data_[i];
    }
    return sum;
  }

  constexpr bool operator==(const Matrix& other) const {
    for (size_t i = 0; i < Size; ++i) {
      if (data_[i] != other.data_[i]) return false;
    }
    return true;
  }

  constexpr bool operator!=(const Matrix& other) const {
    return !operator==(other);
  }

  // AdditiveSemigroup methods
  constexpr Matrix Add(const Matrix& other) const {
    Matrix ret;
    for (size_t i = 0; i < Size; ++i) {
      ret.data_[i] = data_[i] + other.data_[i];
    }
    return ret;
  }

  constexpr Matrix& AddInPlace(const Matrix& other) {
    for (size_t i = 0; i < Size; ++i) {
      data_[i] += other.data_[i];
    }
    return *this;
  }

  constexpr Matrix& DoubleInPlace() {
    for (size_t i = 0; i < Size; ++i) {
      data_[i] += data_[i];
    }
    return *this;
  }

  // AdditiveGroup methods
  constexpr Matrix Sub(const Matrix& other) const {
    Matrix ret;
    for (size_t i = 0; i < Size; ++i) {
      ret.data_[i] = data_[i] - other.data_[i];
    }
    return ret;
  }

  constexpr Matrix& SubInPlace(const Matrix& other) {
    for (size_t i = 0; i < Size; ++i) {
      data_[i] -= other.data_[i];
    }
    return *this;
  }

  constexpr Matrix& NegInPlace() {
    for (size_t i = 0; i < Size; ++i) {
      data_[i] = -data_[i];
    }
    return *this;
  }

  // MultiplicativeSemigroup methods
  template <size_t Cols2>
  constexpr Matrix<T, Rows, Cols2> Mul(
      const Matrix<T, Cols, Cols2>& other) const {
    Matrix<T, Rows, Cols2> ret;
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols2; ++j) {
        for (size_t k = 0; k < Cols; ++k) {
          ret.data_[i * Cols2 + j] +=
              data_[i * Cols + k] * other.data_[k * Cols2 + j];
        }
      }
    }
    return ret;
  }

  template <typename U = T>
  constexpr Matrix Mul(const U& s) const {
    Matrix ret;
    for (size_t i = 0; i < Size; ++i) {
      ret[i] = data_[i] * s;
    }
    return ret;
  }

  template <typename U = T>
  constexpr Matrix& MulInPlace(const U& s) {
    for (size_t i = 0; i < Size; ++i) {
      data_[i++] *= s;
    }
    return *this;
  }

  constexpr Matrix operator/(const T& s) const {
    Matrix ret;
    for (size_t i = 0; i < Size; ++i) {
      ret.data_[i] = data_[i] / s;
    }
    return ret;
  }

  constexpr Matrix& operator/=(const T& s) {
    for (size_t i = 0; i < Size; ++i) {
      data_[i] /= s;
    }
    return *this;
  }

  constexpr MatrixView<T> AsMatrixView() const { return Block(0, 0); }

  constexpr MatrixView<T> Block(size_t row, size_t col, size_t rows = 0,
                                size_t cols = 0) const {
    DCHECK_LT(row, Rows);
    DCHECK_LT(col, Cols);
    if (rows == 0) {
      rows = Rows - row;
    } else {
      DCHECK_LE(rows, Rows - row);
    }
    if (cols == 0) {
      cols = Cols - col;
    } else {
      DCHECK_LE(cols, Cols - col);
    }
    return {&data_[row * Cols + col], rows, cols, Cols};
  }

  constexpr Matrix<T, Cols, Rows> Transpose() const {
    Matrix<T, Cols, Rows> ret;
    for (size_t i = 0; i < Rows; ++i) {
      for (size_t j = 0; j < Cols; ++j) {
        ret.at(i, j) = at(j, i);
      }
    }
    return ret;
  }

 protected:
  template <typename T2, size_t Rows2_, size_t Cols2_>
  friend class Matrix;

  T data_[Size] = {
      math::Zero<T>(),
  };
};

template <typename T, size_t Rows, size_t Cols>
std::ostream& operator<<(std::ostream& os,
                         const Matrix<T, Rows, Cols>& matrix) {
  return os << matrix.ToString();
}

template <typename T, size_t Rows, size_t Cols>
constexpr Matrix<T, Rows, Cols> operator*(T a,
                                          const Matrix<T, Rows, Cols>& matrix) {
  return matrix * a;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0) {
  static_assert(Size == 1, "Invalid matrix size");
  data_[0] = v0;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0, T v1) {
  static_assert(Size == 2, "Invalid matrix size");
  data_[0] = v0;
  data_[1] = v1;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0, T v1, T v2) {
  static_assert(Size == 3, "Invalid matrix size");
  data_[0] = v0;
  data_[1] = v1;
  data_[2] = v2;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0, T v1, T v2, T v3) {
  static_assert(Size == 4, "Invalid matrix size");
  data_[0] = v0;
  data_[1] = v1;
  data_[2] = v2;
  data_[3] = v3;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0, T v1, T v2, T v3, T v4) {
  static_assert(Size == 5, "Invalid matrix size");
  data_[0] = v0;
  data_[1] = v1;
  data_[2] = v2;
  data_[3] = v3;
  data_[4] = v4;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0, T v1, T v2, T v3, T v4, T v5) {
  static_assert(Size == 6, "Invalid matrix size");
  data_[0] = v0;
  data_[1] = v1;
  data_[2] = v2;
  data_[3] = v3;
  data_[4] = v4;
  data_[5] = v5;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0, T v1, T v2, T v3, T v4, T v5,
                                          T v6) {
  static_assert(Size == 7, "Invalid matrix size");
  data_[0] = v0;
  data_[1] = v1;
  data_[2] = v2;
  data_[3] = v3;
  data_[4] = v4;
  data_[5] = v5;
  data_[6] = v6;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0, T v1, T v2, T v3, T v4, T v5,
                                          T v6, T v7) {
  static_assert(Size == 8, "Invalid matrix size");
  data_[0] = v0;
  data_[1] = v1;
  data_[2] = v2;
  data_[3] = v3;
  data_[4] = v4;
  data_[5] = v5;
  data_[6] = v6;
  data_[7] = v7;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0, T v1, T v2, T v3, T v4, T v5,
                                          T v6, T v7, T v8) {
  static_assert(Size == 9, "Invalid matrix size");
  data_[0] = v0;
  data_[1] = v1;
  data_[2] = v2;
  data_[3] = v3;
  data_[4] = v4;
  data_[5] = v5;
  data_[6] = v6;
  data_[7] = v7;
  data_[8] = v8;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0, T v1, T v2, T v3, T v4, T v5,
                                          T v6, T v7, T v8, T v9) {
  static_assert(Size == 10, "Invalid matrix size");
  data_[0] = v0;
  data_[1] = v1;
  data_[2] = v2;
  data_[3] = v3;
  data_[4] = v4;
  data_[5] = v5;
  data_[6] = v6;
  data_[7] = v7;
  data_[8] = v8;
  data_[9] = v9;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0, T v1, T v2, T v3, T v4, T v5,
                                          T v6, T v7, T v8, T v9, T v10,
                                          T v11) {
  static_assert(Size == 12, "Invalid matrix size");
  data_[0] = v0;
  data_[1] = v1;
  data_[2] = v2;
  data_[3] = v3;
  data_[4] = v4;
  data_[5] = v5;
  data_[6] = v6;
  data_[7] = v7;
  data_[8] = v8;
  data_[9] = v9;
  data_[10] = v10;
  data_[11] = v11;
}

template <typename T, size_t Rows_, size_t Cols_>
constexpr Matrix<T, Rows_, Cols_>::Matrix(T v0, T v1, T v2, T v3, T v4, T v5,
                                          T v6, T v7, T v8, T v9, T v10, T v11,
                                          T v12, T v13, T v14, T v15) {
  static_assert(Size == 16, "Invalid matrix size");
  data_[0] = v0;
  data_[1] = v1;
  data_[2] = v2;
  data_[3] = v3;
  data_[4] = v4;
  data_[5] = v5;
  data_[6] = v6;
  data_[7] = v7;
  data_[8] = v8;
  data_[9] = v9;
  data_[10] = v10;
  data_[11] = v11;
  data_[12] = v12;
  data_[13] = v13;
  data_[14] = v14;
  data_[15] = v15;
}

template <typename T, size_t Rows_, size_t Cols_>
struct MatrixTraits<Matrix<T, Rows_, Cols_>> {
 public:
  static constexpr size_t Rows = Rows_;
  static constexpr size_t Cols = Cols_;
  static constexpr size_t Size = Rows * Cols;

  static constexpr bool is_view = false;

  typedef T value_type;
};

template <typename T, size_t Rows, size_t Cols>
class MultiplicativeIdentity<Matrix<T, Rows, Cols>> {
 public:
  using M = Matrix<T, Rows, Cols>;

  static const M& One() {
    static base::NoDestructor<M> one(M::Identity());
    return *one;
  }

  constexpr static bool IsOne(const M& value) { return value.IsIdentity(); }
};

template <typename T, size_t Rows, size_t Cols>
class AdditiveIdentity<Matrix<T, Rows, Cols>> {
 public:
  using M = Matrix<T, Rows, Cols>;

  static const M& Zero() {
    static base::NoDestructor<M> zero(M::Zero());
    return *zero;
  }

  constexpr static bool IsZero(const M& value) { return value.IsZero(); }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_MATRIX_MATRIX_H_