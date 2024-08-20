// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_NAIVE_BATCH_FFT_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_NAIVE_BATCH_FFT_H_

#include <utility>
#include <vector>

#include "tachyon/base/bits.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/math/polynomials/univariate/two_adic_subgroup.h"

namespace tachyon::math {

template <typename F>
class NaiveBatchFFT : public TwoAdicSubgroup<NaiveBatchFFT<F>> {
 public:
  template <typename Derived>
  void FFTBatch(Eigen::MatrixBase<Derived>& mat) const {
    size_t rows = mat.rows();
    size_t cols = mat.cols();
    CHECK(base::bits::IsPowerOfTwo(rows));
    F g;
    CHECK(F::GetRootOfUnity(rows, &g));

    RowMajorMatrix<F> res = RowMajorMatrix<F>::Zero(rows, cols);

    base::Parallelize(
        rows, [rows, cols, &res, &mat, &g](size_t len, size_t chunk_offset,
                                           size_t chunk_size) {
          size_t src_row = chunk_offset * chunk_size;
          F base_pow = g.Pow(src_row);
          // NOTE (chokobole): |base_pow| is multiplied one extra time for the
          // sake of code readability, but we choose to overlook this for
          // simplicity.
          for (size_t res_r = src_row; res_r < src_row + len; ++res_r) {
            // NOTE(chokobole): |rows| is guaranteed to be positive number
            // because of the above |CHECK(base::bits::IsPowerOfTwo(rows))|.
            F pow = F::One();
            for (size_t src_r = 0; src_r < rows - 1; ++src_r) {
              for (size_t col = 0; col < cols; ++col) {
                res(res_r, col) += pow * mat(src_r, col);
              }
              pow *= base_pow;
            }
            for (size_t col = 0; col < cols; ++col) {
              res(res_r, col) += pow * mat(rows - 1, col);
            }
            base_pow *= g;
          }
        });

    mat = std::move(res);
  }

  // Compute the low-degree extension of each column in |mat| onto a coset of
  // a larger subgroup.
  template <typename Derived>
  RowMajorMatrix<F> CosetLDEBatch(Eigen::MatrixBase<Derived>& mat,
                                  size_t added_bits, F shift) const {
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    this->IFFTBatch(mat);
    Eigen::Index rows = mat.rows();
    Eigen::Index new_rows = rows << added_bits;
    Eigen::Index cols = mat.cols();

    // Possible crash if the new resized length overflows
    RowMajorMatrix<F> ret(new_rows, cols);
    OMP_PARALLEL_FOR(Eigen::Index row = 0; row < new_rows; ++row) {
      if (row < rows) {
        ret.row(row) = mat.row(row);
      } else {
        ret.row(row).setZero();
      }
    }
    CosetFFTBatch(ret, shift);
    return ret;
  }

 private:
  // Compute the "coset DFT" of each column in |mat|. This can be viewed as
  // interpolation onto a coset of a multiplicative subgroup, rather than the
  // subgroup itself.
  void CosetFFTBatch(Eigen::MatrixBase<RowMajorMatrix<F>>& mat, F shift) const {
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    // Observe that
    // yᵢ = ∑ⱼ cⱼ (s gⁱ)ʲ
    //    = ∑ⱼ (cⱼ sʲ) (gⁱ)ʲ
    // which has the structure of an ordinary DFT, except each coefficient cⱼ
    // is first replaced by cⱼ s.
    size_t rows = mat.rows();
    base::Parallelize(
        rows, [&mat, &shift](Eigen::Index len, Eigen::Index chunk_offset,
                             Eigen::Index chunk_size) {
          Eigen::Index start = chunk_offset * chunk_size;
          F weight = shift.Pow(start);
          // NOTE: It is not possible to have empty chunk so this is safe
          for (Eigen::Index row = start; row < start + len - 1; ++row) {
            for (Eigen::Index col = 0; col < mat.cols(); ++col) {
              mat(row, col) *= weight;
            }
            weight *= shift;
          }
          for (Eigen::Index col = 0; col < mat.cols(); ++col) {
            mat(start + len - 1, col) *= weight;
          }
        });
    FFTBatch(mat);
  }
};

template <typename F>
struct TwoAdicSubgroupTraits<NaiveBatchFFT<F>> {
  using Field = F;
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_NAIVE_BATCH_FFT_H_
