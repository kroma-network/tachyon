#ifndef TACHYON_MATH_MATRIX_MATRIX_UTILS_H_
#define TACHYON_MATH_MATRIX_MATRIX_UTILS_H_

#include <numeric>
#include <utility>
#include <vector>

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/base/bits.h"
#include "tachyon/base/compiler_specific.h"
#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/profiler.h"
#include "tachyon/math/finite_fields/extended_packed_field_traits_forward.h"
#include "tachyon/math/finite_fields/extension_field_traits_forward.h"
#include "tachyon/math/finite_fields/finite_field_traits.h"
#include "tachyon/math/finite_fields/packed_field_traits_forward.h"
#include "tachyon/math/matrix/matrix_types.h"

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

// Creates a vector of packed fields for a given matrix row. If the length
// of the row is not a multiple of |PackedField::N|, the last |PackedField|
// element populates leftover values with |F::Zero()|.
template <typename Expr, int BlockRows, int BlockCols, bool InnerPanel,
          typename PackedField =
              typename PackedFieldTraits<typename Expr::Scalar>::PackedField>
std::vector<PackedField> PackRowHorizontallyPadded(
    const Eigen::Block<Expr, BlockRows, BlockCols, InnerPanel>& matrix_row) {
  using F = typename FiniteFieldTraits<PackedField>::PrimeField;

  size_t cols = static_cast<size_t>(matrix_row.cols());
  size_t size = (cols + PackedField::N - 1) / PackedField::N;
  return base::CreateVector(size, [cols, &matrix_row](size_t i) {
    size_t col = i * PackedField::N;
    return PackedField::From([col, cols, &matrix_row](size_t c) {
      if (LIKELY(col + c < cols)) {
        return matrix_row[col + c];
      } else {
        return F::Zero();
      }
    });
  });
}

// Packs |PackedField::N| rows, starting at the given row index. Each
// |PackedField::N| selected elements in a column are converted into a
// PackedField. Rows included wrap around to the starting index 0 to fully
// populate the packed fields.
template <typename PackedField, typename Derived>
std::vector<PackedField> PackRowVertically(
    const Eigen::MatrixBase<Derived>& matrix, size_t row) {
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  if constexpr (FiniteFieldTraits<Scalar>::kIsExtensionField) {
    constexpr size_t kDegree =
        ExtensionFieldTraits<Scalar>::kDegreeOverBasePrimeField;
    size_t num = matrix.cols() * kDegree;
    return base::CreateVector(num, [row, &matrix](size_t n) {
      size_t col = n / kDegree;
      size_t idx = n - col * kDegree;
      return PackedField::From([row, col, idx, &matrix](size_t i) {
        return matrix((row + i) % matrix.rows(), col)[idx];
      });
    });
  } else {
    return base::CreateVector(matrix.cols(), [row, &matrix](size_t col) {
      return PackedField::From([row, col, &matrix](size_t i) {
        return matrix((row + i) % matrix.rows(), col);
      });
    });
  }
}

// Swaps rows of a |Eigen::MatrixBase| such that each row is changed to the row
// accessed with the reversed bits of the current index. Crashes if the number
// of rows is not a power of two.
template <typename Derived>
#if HAS_ATTRIBUTE(optimize)
void __attribute__((optimize(3)))
ReverseMatrixIndexBits(Eigen::MatrixBase<Derived>& mat) {
#else
void ReverseMatrixIndexBits(Eigen::MatrixBase<Derived>& mat) {
#endif
  TRACE_EVENT("Utils", "MatrixUtils::ReverseMatrixIndexBits");

  static_assert(Derived::IsRowMajor);

  size_t rows = static_cast<size_t>(mat.rows());
  if (rows == 0) {
    return;
  }
  uint32_t log_n = base::bits::CheckedLog2(rows);

  OMP_PARALLEL_FOR(size_t row = 1; row < rows; ++row) {
    size_t ridx = base::bits::ReverseBitsLen(row, log_n);
    if (row < ridx) {
      absl::Span<uint8_t> row1(
          reinterpret_cast<uint8_t*>(mat.derived().data() + row * mat.cols()),
          mat.cols() * sizeof(typename Derived::Scalar));
      absl::Span<uint8_t> row2(
          reinterpret_cast<uint8_t*>(mat.derived().data() + ridx * mat.cols()),
          mat.cols() * sizeof(typename Derived::Scalar));

      std::swap_ranges(row1.begin(), row1.end(), row2.begin());
    }
  }
}

// Returns a vector of size |num_chunks| that holds submatrices with
// |num_chunks| amount of rows of a given matrix |mat| using vertical
// striding.
template <typename Derived>
std::vector<Eigen::Block<Derived>> SplitMat(Eigen::Index num_chunks,
                                            Eigen::MatrixBase<Derived>& mat) {
  Eigen::Index total_span = mat.rows() - num_chunks + 1;
  CHECK_GE(mat.rows(), num_chunks);
  Eigen::Index num_cols = mat.cols();
  return base::CreateVector(num_chunks, [&mat, total_span, num_cols](size_t i) {
    return mat.block(i, 0, total_span, num_cols);
  });
}

template <typename ExtField, typename Derived>
std::vector<ExtField> DotExtPowers(const Eigen::MatrixBase<Derived>& mat,
                                   const ExtField& base) {
  using F = typename ExtensionFieldTraits<ExtField>::BaseField;
  using PackedField = typename PackedFieldTraits<F>::PackedField;
  using ExtendedPackedField =
      typename ExtendedPackedFieldTraits<ExtField>::ExtendedPackedField;
  Eigen::Index rows = mat.rows();
  size_t packed_n = PackedField::N;
  std::vector<ExtendedPackedField> packed_ext_powers =
      ExtField::GetExtendedPackedPowers(
          base, ((static_cast<size_t>(mat.cols()) + packed_n - 1) / packed_n) *
                    packed_n);
  return base::CreateVectorParallel(rows, [&mat,
                                           &packed_ext_powers](Eigen::Index r) {
    std::vector<PackedField> row_packed = PackRowHorizontallyPadded(mat.row(r));
    ExtendedPackedField packed_sum_of_packed(ExtendedPackedField::Zero());
    for (size_t i = 0; i < row_packed.size(); ++i) {
      packed_sum_of_packed += packed_ext_powers[i] * row_packed[i];
    }
    ExtField ret(ExtField::Zero());
    for (size_t i = 0; i < ExtField::ExtensionDegree(); ++i) {
      const PackedField& packed = packed_sum_of_packed[i];
      for (size_t j = 0; j < PackedField::N; ++j) {
        ret[i] += packed[j];
      }
    }
    return ret;
  });
}

template <typename Derived>
Eigen::Map<Derived> Map(Eigen::PlainObjectBase<Derived>& mat) {
  return Eigen::Map<Derived>(mat.data(), mat.rows(), mat.cols());
}

template <typename Derived>
Eigen::Map<const Derived> Map(const Eigen::PlainObjectBase<Derived>& mat) {
  return Eigen::Map<const Derived>(mat.data(), mat.rows(), mat.cols());
}

template <typename Derived, typename Scalar = typename Derived::Scalar,
          Eigen::Index Rows = Derived::RowsAtCompileTime,
          Eigen::Index Cols = Derived::ColsAtCompileTime>
std::array<std::array<Scalar, Cols>, Rows> To2DArray(
    const Eigen::MatrixBase<Derived>& matrix) {
  std::array<std::array<Scalar, Cols>, Rows> ret;
  for (Eigen::Index r = 0; r < Rows; ++r) {
    for (Eigen::Index c = 0; c < Cols; ++c) {
      ret[r][c] = matrix(r, c);
    }
  }
  return ret;
}

template <typename Derived, typename Scalar = typename Derived::Scalar,
          Eigen::Index N = Derived::RowsAtCompileTime == 1
                               ? Derived::ColsAtCompileTime
                               : Derived::RowsAtCompileTime>
std::array<Scalar, N> ToArray(const Eigen::MatrixBase<Derived>& vector) {
  std::array<Scalar, N> ret;
  for (Eigen::Index i = 0; i < N; ++i) {
    ret[i] = vector[i];
  }
  return ret;
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_MATRIX_MATRIX_UTILS_H_
