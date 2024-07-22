#ifndef TACHYON_MATH_MATRIX_MATRIX_OPERATIONS_H_
#define TACHYON_MATH_MATRIX_MATRIX_OPERATIONS_H_

#include "tachyon/base/openmp_util.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::math {

// The motivation of the variants of |MulMatVec()| is as follows:

// NOTE(chokobole): Eigen matrix multiplication has a computational
// overhead unlike naive matrix multiplication.
//
// Example: Multiplying a 2x2 matrix by a 2x1 vector:
//
// +----+----+   +----+
// | m₀ | m₁ | * | v₀ |
// +----+----+   +----+
// | m₂ | m₃ |   | v₁ |
// +----+----+   +----+
//
// The operations involved in this multiplication are as follows:
//
// 1 * 1
// 1 * 1
// m₀ * v₀
// m₀v₀ + 0
// m₂ * v₀
// m₂v₀ + 0
// m₁ * v₁
// m₁v₁ + m₀v₀
// m₃ * v₁
// m₃v₁ + m₂v₀
// m₁v₁ + m₀v₀ * 1
// m₁v₁ + m₀v₀ + 0
// m₃v₁ + m₂v₀ * 1
// m₃v₁ + m₂v₀ + 0

template <typename Derived, typename Derived2,
          typename F = typename Derived::Scalar>
math::Vector<F> MulMatVec(const Eigen::MatrixBase<Derived>& matrix,
                          const Eigen::MatrixBase<Derived2>& vector) {
  static_assert(std::is_same_v<F, typename Derived2::Scalar>);

  math::Vector<F> ret = math::Vector<F>::Constant(vector.size(), F::Zero());
  OPENMP_PARALLEL_FOR(Eigen::Index i = 0; i < matrix.rows(); ++i) {
    for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
      ret[i] += matrix(i, j) * vector[j];
    }
  }
  return ret;
}

template <typename Derived, typename Derived2,
          typename F = typename Derived::Scalar>
math::Vector<F> MulMatVecSerial(const Eigen::MatrixBase<Derived>& matrix,
                                const Eigen::MatrixBase<Derived2>& vector) {
  static_assert(std::is_same_v<F, typename Derived2::Scalar>);

  math::Vector<F> ret = math::Vector<F>::Constant(vector.size(), F::Zero());
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
      ret[i] += matrix(i, j) * vector[j];
    }
  }
  return ret;
}

template <typename Derived, typename Derived2, enum Eigen::AccessorLevels Level,
          typename F = typename Derived::Scalar>
math::Vector<F> MulMatVec(
    const Eigen::MatrixBase<Derived>& matrix,
    const Eigen::DenseCoeffsBase<Derived2, Level>& vector) {
  static_assert(std::is_same_v<F, typename Derived2::Scalar>);

  math::Vector<F> ret = math::Vector<F>::Constant(vector.size(), F::Zero());
  if (vector.rows() == 1) {
    OPENMP_PARALLEL_FOR(Eigen::Index i = 0; i < matrix.rows(); ++i) {
      for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
        ret[i] += matrix(i, j) * vector(0, j);
      }
    }
  } else if (vector.cols() == 1) {
    OPENMP_PARALLEL_FOR(Eigen::Index i = 0; i < matrix.rows(); ++i) {
      for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
        ret[i] += matrix(i, j) * vector(j, 0);
      }
    }
  } else {
    NOTREACHED();
  }
  return ret;
}

template <typename Derived, typename Derived2, enum Eigen::AccessorLevels Level,
          typename F = typename Derived::Scalar>
math::Vector<F> MulMatVecSerial(
    const Eigen::MatrixBase<Derived>& matrix,
    const Eigen::DenseCoeffsBase<Derived2, Level>& vector) {
  static_assert(std::is_same_v<F, typename Derived2::Scalar>);

  math::Vector<F> ret = math::Vector<F>::Constant(vector.size(), F::Zero());
  if (vector.rows() == 1) {
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
      for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
        ret[i] += matrix(i, j) * vector(0, j);
      }
    }
  } else if (vector.cols() == 1) {
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
      for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
        ret[i] += matrix(i, j) * vector(j, 0);
      }
    }
  } else {
    NOTREACHED();
  }
  return ret;
}

template <typename Derived, typename Derived2,
          typename F = typename Derived::Scalar>
math::Matrix<F> MulMatMat(const Eigen::MatrixBase<Derived>& matrix,
                          const Eigen::MatrixBase<Derived2>& matrix2) {
  static_assert(std::is_same_v<F, typename Derived2::Scalar>);
  CHECK_EQ(matrix.cols(), matrix2.rows());

  math::Matrix<F> ret =
      math::Matrix<F>::Constant(matrix.rows(), matrix2.cols(), F::Zero());
  OPENMP_PARALLEL_FOR(Eigen::Index i = 0; i < matrix.rows(); ++i) {
    for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
      for (Eigen::Index k = 0; k < matrix2.cols(); ++k) {
        ret(i, k) += matrix(i, j) * matrix2(j, k);
      }
    }
  }
  return ret;
}

template <typename Derived, typename Derived2,
          typename F = typename Derived::Scalar>
math::Matrix<F> MulMatMatSerial(const Eigen::MatrixBase<Derived>& matrix,
                                const Eigen::MatrixBase<Derived2>& matrix2) {
  static_assert(std::is_same_v<F, typename Derived2::Scalar>);
  CHECK_EQ(matrix.cols(), matrix2.rows());

  math::Matrix<F> ret =
      math::Matrix<F>::Constant(matrix.rows(), matrix2.cols(), F::Zero());
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
      for (Eigen::Index k = 0; k < matrix2.cols(); ++k) {
        ret(i, k) += matrix(i, j) * matrix2(j, k);
      }
    }
  }
  return ret;
}
}  // namespace tachyon::math

#endif  // TACHYON_MATH_MATRIX_MATRIX_OPERATIONS_H_
