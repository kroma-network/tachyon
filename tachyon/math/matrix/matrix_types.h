#ifndef TACHYON_MATH_MATRIX_MATRIX_TYPES_H_
#define TACHYON_MATH_MATRIX_MATRIX_TYPES_H_

#include "Eigen/Core"

namespace tachyon::math {

template <typename PrimeFieldTy>
using Matrix = Eigen::Matrix<PrimeFieldTy, Eigen::Dynamic, Eigen::Dynamic>;

template <typename PrimeFieldTy>
using Vector = Eigen::Matrix<PrimeFieldTy, Eigen::Dynamic, 1>;

template <typename PrimeFieldTy>
using RowVector = Eigen::Matrix<PrimeFieldTy, 1, Eigen::Dynamic>;

}  // namespace tachyon::math

#endif  // TACHYON_MATH_MATRIX_MATRIX_TYPES_H_
