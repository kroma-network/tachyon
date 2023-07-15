#ifndef TACHYON_MATH_MATRIX_MATRIX_INVERSE_OPERATOR_H_
#define TACHYON_MATH_MATRIX_MATRIX_INVERSE_OPERATOR_H_

#include <type_traits>

#include "tachyon/math/matrix/matrix_traits.h"

namespace tachyon {
namespace math {
namespace internal {

namespace {

template <size_t N, typename T, std::enable_if_t<N == 2>* = nullptr>
constexpr T FastDeterminant(T* data) {
  return data[0] * data[3] - data[1] * data[2];
}

template <size_t N, typename T, std::enable_if_t<N == 3>* = nullptr>
constexpr T FastDeterminant(T* data) {
  T d0 = data[4] * data[8] - data[5] * data[7];
  T d1 = data[3] * data[8] - data[5] * data[6];
  T d2 = data[3] * data[7] - data[4] * data[6];
  return data[0] * d0 - data[1] * d1 + data[2] * d2;
}

template <size_t N, typename T, std::enable_if_t<N == 4>* = nullptr>
constexpr T FastDeterminant(T* data) {
  T d80 = data[8] * data[13] - data[9] * data[12];
  T d81 = data[8] * data[14] - data[10] * data[12];
  T d82 = data[8] * data[15] - data[11] * data[12];
  T d90 = data[9] * data[14] - data[10] * data[13];
  T d91 = data[9] * data[15] - data[11] * data[13];
  T da0 = data[10] * data[15] - data[11] * data[14];

  T d0 = data[5] * da0 - data[6] * d91 + data[7] * d90;
  T d1 = data[4] * da0 - data[6] * d82 + data[7] * d81;
  T d2 = data[4] * d91 - data[5] * d82 + data[7] * d80;
  T d3 = data[4] * d90 - data[5] * d81 + data[6] * d80;

  return data[0] * d0 - data[1] * d1 + data[2] * d2 - data[3] * d3;
}

template <size_t N, typename T, std::enable_if_t<N == 2>* = nullptr>
constexpr void FastInverse(const T* src, T* dst) {
  T det = src[0] * src[3] - src[1] * src[2];

  dst[0] = src[3] / det;
  dst[1] = -src[1] / det;
  dst[2] = -src[2] / det;
  dst[3] = src[0] / det;
}

template <size_t N, typename T, std::enable_if_t<N == 3>* = nullptr>
constexpr void FastInverse(const T* src, T* dst) {
  T d0 = src[4] * src[8] - src[5] * src[7];
  T d1 = src[3] * src[8] - src[5] * src[6];
  T d2 = src[3] * src[7] - src[4] * src[6];
  T d3 = src[1] * src[8] - src[2] * src[7];
  T d4 = src[0] * src[8] - src[2] * src[6];
  T d5 = src[0] * src[7] - src[1] * src[6];
  T d6 = src[1] * src[5] - src[2] * src[4];
  T d7 = src[0] * src[5] - src[2] * src[3];
  T d8 = src[0] * src[4] - src[1] * src[3];

  T det = src[0] * d0 - src[1] * d1 + src[2] * d2;

  dst[0] = d0 / det;
  dst[1] = -d3 / det;
  dst[2] = d6 / det;
  dst[3] = -d1 / det;
  dst[4] = d4 / det;
  dst[5] = -d7 / det;
  dst[6] = d2 / det;
  dst[7] = -d5 / det;
  dst[8] = d8 / det;
}

template <size_t N, typename T, std::enable_if_t<N == 4>* = nullptr>
constexpr void FastInverse(const T* src, T* dst) {
  T d40 = src[4] * src[9] - src[5] * src[8];
  T d41 = src[4] * src[10] - src[6] * src[8];
  T d42 = src[4] * src[11] - src[7] * src[8];
  T d43 = src[4] * src[13] - src[5] * src[12];
  T d44 = src[4] * src[14] - src[6] * src[12];
  T d45 = src[4] * src[15] - src[7] * src[12];
  T d50 = src[5] * src[10] - src[6] * src[9];
  T d51 = src[5] * src[11] - src[7] * src[9];
  T d52 = src[5] * src[14] - src[6] * src[13];
  T d53 = src[5] * src[15] - src[7] * src[13];
  T d60 = src[6] * src[11] - src[7] * src[10];
  T d61 = src[6] * src[15] - src[7] * src[14];
  T d80 = src[8] * src[13] - src[9] * src[12];
  T d81 = src[8] * src[14] - src[10] * src[12];
  T d82 = src[8] * src[15] - src[11] * src[12];
  T d90 = src[9] * src[14] - src[10] * src[13];
  T d91 = src[9] * src[15] - src[11] * src[13];
  T da0 = src[10] * src[15] - src[11] * src[14];

  T d0 = src[5] * da0 - src[6] * d91 + src[7] * d90;
  T d1 = src[4] * da0 - src[6] * d82 + src[7] * d81;
  T d2 = src[4] * d91 - src[5] * d82 + src[7] * d80;
  T d3 = src[4] * d90 - src[5] * d81 + src[6] * d80;
  T d4 = src[1] * da0 - src[2] * d91 + src[3] * d90;
  T d5 = src[0] * da0 - src[2] * d82 + src[3] * d81;
  T d6 = src[0] * d91 - src[1] * d82 + src[3] * d80;
  T d7 = src[0] * d90 - src[1] * d81 + src[2] * d80;
  T d8 = src[1] * d61 - src[2] * d53 + src[3] * d52;
  T d9 = src[0] * d61 - src[2] * d45 + src[3] * d44;
  T da = src[0] * d53 - src[1] * d45 + src[3] * d43;
  T db = src[0] * d52 - src[1] * d44 + src[2] * d43;
  T dc = src[1] * d60 - src[2] * d51 + src[3] * d50;
  T dd = src[0] * d60 - src[2] * d42 + src[3] * d41;
  T de = src[0] * d51 - src[1] * d42 + src[3] * d40;
  T df = src[0] * d50 - src[1] * d41 + src[2] * d40;

  T det = src[0] * d0 - src[1] * d1 + src[2] * d2 - src[3] * d3;

  dst[0] = d0 / det;
  dst[1] = -d4 / det;
  dst[2] = d8 / det;
  dst[3] = -dc / det;
  dst[4] = -d1 / det;
  dst[5] = d5 / det;
  dst[6] = -d9 / det;
  dst[7] = dd / det;
  dst[8] = d2 / det;
  dst[9] = -d6 / det;
  dst[10] = da / det;
  dst[11] = -de / det;
  dst[12] = -d3 / det;
  dst[13] = d7 / det;
  dst[14] = -db / det;
  dst[15] = df / det;
}

}  // namespace

template <template <typename T, size_t, size_t> class Matrix, typename T,
          size_t Rows, size_t Cols,
          std::enable_if_t<(Rows == Cols && Rows <= 4)>* = nullptr>
constexpr T DeterminantImpl(const Matrix<T, Rows, Cols>* matrix) {
  return FastDeterminant<Rows>(matrix->data());
}

// TODO(chokobole): This is super slow, don't use this for production code!
template <template <typename T, size_t, size_t> class Matrix, typename T,
          size_t Rows, size_t Cols,
          std::enable_if_t<(Rows == Cols && Rows > 4)>* = nullptr>
constexpr T DeterminantImpl(const Matrix<T, Rows, Cols>* matrix) {
  T ret = T::Zero();
  T sign = T::One();
  Matrix<T, Rows - 1, Cols - 1> mat_copy;
  T* data_copy = mat_copy.data();
  for (size_t i = 0; i < Cols; ++i) {
    size_t data_copy_idx = 0;
    for (size_t j = 1; j < Rows; ++j) {
      for (size_t k = 0; k < Cols; ++k) {
        if (i == k) continue;
        data_copy[data_copy_idx++] = matrix->at(j, k);
      }
    }
    ret += sign * matrix->at(0, i) * DeterminantImpl(&mat_copy);
    sign.NegInPlace();
  }
  return ret;
}

template <template <typename T, size_t, size_t> class Matrix, typename T,
          size_t Rows, size_t Cols,
          std::enable_if_t<(Rows == Cols && Rows <= 4)>* = nullptr>
constexpr Matrix<T, Rows, Cols> InverseImpl(
    const Matrix<T, Rows, Cols>* matrix) {
  Matrix<T, Rows, Cols> ret;
  FastInverse<Rows>(matrix->data(), ret.data());
  return ret;
}

// TODO(chokobole): This is super slow, don't use this for production code!
template <template <typename T, size_t, size_t> class Matrix, typename T,
          size_t Rows, size_t Cols,
          std::enable_if_t<(Rows == Cols && Rows > 4)>* = nullptr>
constexpr Matrix<T, Rows, Cols> InverseImpl(
    const Matrix<T, Rows, Cols>* matrix) {
  Matrix<T, Rows, Cols> ret;
  T det = matrix->Determinant();
  Matrix<T, Rows - 1, Cols - 1> mat_copy;
  T* data_copy = mat_copy.data();
  for (size_t i = 0; i < Rows; ++i) {
    for (size_t j = 0; j < Cols; ++j) {
      size_t data_copy_idx = 0;
      for (size_t r = 0; r < Rows; ++r) {
        if (i == r) continue;
        for (size_t c = 0; c < Cols; ++c) {
          if (j == c) continue;
          data_copy[data_copy_idx++] = matrix->at(r, c);
        }
      }
      T sign = T::One();
      if ((i + j) % 2 != 0) {
        sign.NegInPlace();
      }
      ret.at(j, i) = sign * DeterminantImpl(&mat_copy) / det;
    }
  }
  return ret;
}

}  // namespace internal

template <typename Matrix>
class MatInverseOperator {
 public:
  constexpr static size_t Rows = MatrixTraits<Matrix>::Rows;
  constexpr static size_t Cols = MatrixTraits<Matrix>::Cols;

  using value_type = typename MatrixTraits<Matrix>::value_type;

  constexpr value_type Determinant() const {
    static_assert(Rows == Cols,
                  "Determinant is only defined for square matrix");
    const Matrix* matrix = static_cast<const Matrix*>(this);
    return internal::DeterminantImpl(matrix);
  }

  constexpr Matrix Inverse() const {
    static_assert(Rows == Cols, "Inverse is only defined for square matrix");
    static_assert(!MatrixTraits<Matrix>::is_view,
                  "Can't create new Matrix from MatrixView");
    const Matrix* matrix = static_cast<const Matrix*>(this);
    return internal::InverseImpl(matrix);
  }
};

}  // namespace math
}  // namespace tachyon

#endif  // TACHYON_MATH_MATRIX_MATRIX_INVERSE_OPERATOR_H_