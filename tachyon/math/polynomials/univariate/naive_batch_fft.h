// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_NAIVE_BATCH_FFT_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_NAIVE_BATCH_FFT_H_

#include <vector>

#include "tachyon/base/bits.h"
#include "tachyon/math/polynomials/univariate/two_adic_subgroup.h"

namespace tachyon::math {

template <typename F>
class NaiveBatchFFT : public TwoAdicSubgroup<F> {
 public:
  void FFTBatch(RowMajorMatrix<F>& mat) override {
    Eigen::Index rows = mat.rows();
    Eigen::Index cols = mat.cols();
    CHECK(base::bits::IsPowerOfTwo(rows));
    F g;
    CHECK(F::GetRootOfUnity(rows, &g));

    RowMajorMatrix<F> res = RowMajorMatrix<F>::Zero(rows, cols);

    std::vector<F> points = F::GetSuccessivePowers(rows, g);
    size_t num_points = points.size();

    for (size_t res_r = 0; res_r < num_points; ++res_r) {
      std::vector<F> point_powers = F::GetSuccessivePowers(rows, points[res_r]);
      for (size_t src_r = 0; src_r < num_points; ++src_r) {
        for (Eigen::Index col = 0; col < cols; ++col) {
          res(res_r, col) += point_powers[src_r] * mat(src_r, col);
        }
      }
    }
    mat = res;
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_NAIVE_BATCH_FFT_H_
