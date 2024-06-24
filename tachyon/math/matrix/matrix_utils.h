#ifndef TACHYON_MATH_MATRIX_MATRIX_UTILS_H_
#define TACHYON_MATH_MATRIX_MATRIX_UTILS_H_

#include <vector>

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/math/finite_fields/packed_prime_field_traits_forward.h"

namespace tachyon::math {

// See https://eigen.tuxfamily.org/dox/TopicCustomizing_NullaryExpr.html
template <class ArgType>
struct CirculantHelper {
  typedef Eigen::Matrix<typename ArgType::Scalar, ArgType::SizeAtCompileTime,
                        ArgType::SizeAtCompileTime, Eigen::ColMajor,
                        ArgType::MaxSizeAtCompileTime,
                        ArgType::MaxSizeAtCompileTime>
      MatrixType;
};

template <class ArgType>
class CirculantFunctor {
 public:
  explicit CirculantFunctor(const ArgType& arg) : vec_(arg) {}

  const typename ArgType::Scalar& operator()(Eigen::Index row,
                                             Eigen::Index col) const {
    Eigen::Index index = row - col;
    if (index < 0) index += vec_.size();
    return vec_(index);
  }

 private:
  const ArgType& vec_;
};

template <class ArgType>
Eigen::CwiseNullaryOp<CirculantFunctor<ArgType>,
                      typename CirculantHelper<ArgType>::MatrixType>
MakeCirculant(const Eigen::MatrixBase<ArgType>& arg) {
  typedef typename CirculantHelper<ArgType>::MatrixType MatrixType;
  return MatrixType::NullaryExpr(arg.size(), arg.size(),
                                 CirculantFunctor<ArgType>(arg.derived()));
}

// NOTE(ashjeong): Important! |matrix| should carry the same amount of rows as
// the parent matrix it is a block from. |PackRowHorizontally| currently only
// supports row-major matrices.
template <typename PackedPrimeField, typename Derived, typename PrimeField>
std::vector<PackedPrimeField*> PackRowHorizontally(
    Eigen::Block<Derived>& matrix, size_t row,
    std::vector<PrimeField*>& remaining_values) {
  static_assert(Derived::Options & Eigen::RowMajorBit);
  size_t num_packed = matrix.cols() / PackedPrimeField::N;
  size_t remaining_start_idx = num_packed * PackedPrimeField::N;
  remaining_values = base::CreateVector(
      matrix.cols() - remaining_start_idx,
      [row, remaining_start_idx, &matrix](size_t col) {
        return reinterpret_cast<PrimeField*>(
            matrix.data() + row * matrix.cols() + remaining_start_idx + col);
      });
  return base::CreateVector(num_packed, [row, &matrix](size_t col) {
    return reinterpret_cast<PackedPrimeField*>(
        matrix.data() + row * matrix.cols() + PackedPrimeField::N * col);
  });
}

template <typename PackedPrimeField, typename Derived>
std::vector<PackedPrimeField> PackRowVertically(
    const Eigen::MatrixBase<Derived>& matrix, size_t row) {
  return base::CreateVector(matrix.cols(), [row, &matrix](size_t col) {
    return PackedPrimeField::From([row, col, &matrix](size_t i) {
      return matrix((row + i) % matrix.rows(), col);
    });
  });
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_MATRIX_MATRIX_UTILS_H_
