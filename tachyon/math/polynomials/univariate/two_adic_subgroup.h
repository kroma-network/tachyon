// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_TWO_ADIC_SUBGROUP_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_TWO_ADIC_SUBGROUP_H_

#include <optional>
#include <utility>
#include <vector>

#include "tachyon/base/optional.h"
#include "tachyon/math/matrix/matrix_types.h"
#include "tachyon/math/polynomials/univariate/two_adic_subgroup_traits_forward.h"

namespace tachyon::math {

template <typename Derived>
class TwoAdicSubgroup {
 public:
  using F = typename TwoAdicSubgroupTraits<Derived>::Field;

  // Compute the inverse DFT of each column in |mat|.
  template <typename T>
  void IFFTBatch(Eigen::MatrixBase<T>& mat) const {
    if constexpr (F::Config::kModulusBits > 32) {
      NOTREACHED();
    }
    const Derived* derived = static_cast<const Derived*>(this);
    derived->FFTBatch(mat);
    Eigen::Index rows = mat.rows();
    // TODO(chokobole): Use |size_inv_| instead of directly computing the value.
    F inv = unwrap(F(rows).Inverse());

    mat.row(0) *= inv;
    mat.row(rows / 2) *= inv;
    OMP_PARALLEL_FOR(Eigen::Index row = 1; row < rows / 2; ++row) {
      auto row1 = mat.row(row);
      auto row2 = mat.row(rows - row);
      row1 *= inv;
      row2 *= inv;
      row1.swap(row2);
    }
  }
};

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_TWO_ADIC_SUBGROUP_H_
