// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_TWO_ADIC_SUBGROUP_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_TWO_ADIC_SUBGROUP_H_

#include <optional>
#include <vector>

#include "tachyon/base/optional.h"
#include "tachyon/math/matrix/matrix_types.h"

namespace tachyon::math {

template <typename F>
class TwoAdicSubgroup {
 public:
  virtual ~TwoAdicSubgroup() = default;

  // Compute the discrete Fourier transform (DFT) of each column in |mat|.
  virtual void FFTBatch(RowMajorMatrix<F>& mat) = 0;

  // Compute the inverse DFT of each column in |mat|.
  void IFFTBatch(RowMajorMatrix<F>& mat) {
    static_assert(F::Config::kModulusBits <= 32);
    FFTBatch(mat);
    Eigen::Index rows = mat.rows();
    F inv = unwrap(F(rows).Inverse());

    mat *= inv;

    for (Eigen::Index row = 1; row < rows / 2; ++row) {
      mat.row(row).swap(mat.row(rows - row));
    }
  }

  // Compute the "coset DFT" of each column in |mat|. This can be viewed as
  // interpolation onto a coset of a multiplicative subgroup, rather than the
  // subgroup itself.
  void CosetFFTBatch(RowMajorMatrix<F>& mat, F shift) {
    static_assert(F::Config::kModulusBits <= 32);
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

  // Compute the low-degree extension of each column in |mat| onto a coset of
  // a larger subgroup.
  void CosetLDEBatch(RowMajorMatrix<F>& mat, size_t added_bits, F shift) {
    static_assert(F::Config::kModulusBits <= 32);
    IFFTBatch(mat);
    Eigen::Index rows = mat.rows();
    Eigen::Index cols = mat.cols();

    // Possible crash if the new resized length overflows
    mat.conservativeResizeLike(
        RowMajorMatrix<F>::Zero(rows << added_bits, cols));
    CosetFFTBatch(mat, shift);
  }
};

}  // namespace tachyon::math
#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_TWO_ADIC_SUBGROUP_H_
