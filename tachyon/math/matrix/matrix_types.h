#ifndef TACHYON_MATH_MATRIX_MATRIX_TYPES_H_
#define TACHYON_MATH_MATRIX_MATRIX_TYPES_H_

#include "Eigen/Core"

namespace tachyon::math {

template <typename PrimeField>
using Matrix = Eigen::Matrix<PrimeField, Eigen::Dynamic, Eigen::Dynamic>;

template <typename PrimeField>
using Vector = Eigen::Matrix<PrimeField, Eigen::Dynamic, 1>;

template <typename PrimeField>
using RowVector = Eigen::Matrix<PrimeField, 1, Eigen::Dynamic>;

}  // namespace tachyon::math

#endif  // TACHYON_MATH_MATRIX_MATRIX_TYPES_H_
