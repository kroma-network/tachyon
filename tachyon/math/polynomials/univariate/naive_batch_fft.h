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
class NaiveBatchFFT : public TwoAdicSubgroup<F> {
 public:
  void FFTBatch(Eigen::MatrixBase<RowMajorMatrix<F>>& mat) override {
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
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_NAIVE_BATCH_FFT_H_
