#ifndef TACHYON_MATH_MATRIX_MATRIX_UTILS_H_
#define TACHYON_MATH_MATRIX_MATRIX_UTILS_H_

#include <utility>
#include <vector>

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/base/bits.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/math/finite_fields/extension_field_traits_forward.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"
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
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  if constexpr (FiniteFieldTraits<Scalar>::kIsExtensionField) {
    size_t num =
        matrix.cols() * ExtensionFieldTraits<Scalar>::kDegreeOverBasePrimeField;
    return base::CreateVector(num, [row, &matrix](size_t n) {
      size_t col = n / ExtensionFieldTraits<Scalar>::kDegreeOverBasePrimeField;
      size_t idx =
          n - col * ExtensionFieldTraits<Scalar>::kDegreeOverBasePrimeField;
      return PackedPrimeField::From([row, col, idx, &matrix](size_t i) {
        return matrix((row + i) % matrix.rows(), col)[idx];
      });
    });
  } else {
    return base::CreateVector(matrix.cols(), [row, &matrix](size_t col) {
      return PackedPrimeField::From([row, col, &matrix](size_t i) {
        return matrix((row + i) % matrix.rows(), col);
      });
    });
  }
}

// Expands a |Eigen::MatrixBase|'s rows from |rows| to |rows|^(|added_bits|),
// moving values from row |i| to row |i|^(|added_bits|). All new entries are set
// to |F::Zero()|.
template <typename Derived>
void ExpandInPlaceWithZeroPad(Eigen::MatrixBase<Derived>& mat,
                              size_t added_bits) {
  if (added_bits == 0) {
    return;
  }

  Eigen::Index original_rows = mat.rows();
  Eigen::Index new_rows = mat.rows() << added_bits;
  Eigen::Index cols = mat.cols();

  Derived padded = Derived::Zero(new_rows, cols);

  OPENMP_PARALLEL_FOR(Eigen::Index row = 0; row < original_rows; ++row) {
    Eigen::Index padded_row_index = row << added_bits;
    // TODO(ashjeong): Check if moved properly
    padded.row(padded_row_index) = std::move(mat.row(row));
  }
  mat = std::move(padded);
}

// Swaps rows of a |Eigen::MatrixBase| such that each row is changed to the row
// accessed with the reversed bits of the current index. Crashes if the number
// of rows is not a power of two.
template <typename Derived>
void ReverseMatrixIndexBits(Eigen::MatrixBase<Derived>& mat) {
  size_t rows = static_cast<size_t>(mat.rows());
  if (rows == 0) {
    return;
  }
  CHECK(base::bits::IsPowerOfTwo(rows));
  size_t log_n = base::bits::Log2Ceiling(rows);

  OPENMP_PARALLEL_FOR(size_t row = 1; row < rows; ++row) {
    size_t ridx = base::bits::BitRev(row) >> (sizeof(size_t) * 8 - log_n);
    if (row < ridx) {
      mat.row(row).swap(mat.row(ridx));
    }
  }
}

// Returns a vector of size |num_chunks| that holds submatrices with
// |num_chunks| amount of rows of a given matrix |evals| using vertical
// striding.
template <typename Derived>
std::vector<Eigen::Block<Derived>> SplitEvals(
    size_t num_chunks, Eigen::MatrixBase<Derived>& evals) {
  Eigen::Index total_span = evals.rows() - num_chunks + 1;
  Eigen::Index num_cols = evals.cols();
  return base::CreateVector(num_chunks,
                            [&evals, total_span, num_cols](size_t i) {
                              return evals.block(i, 0, total_span, num_cols);
                            });
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_MATRIX_MATRIX_UTILS_H_
