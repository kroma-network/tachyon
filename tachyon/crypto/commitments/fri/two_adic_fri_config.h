// Copyright (c) 2022 The Plonky3 Authors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.plonky3 and the LICENCE-APACHE.plonky3
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_CONFIG_H_
#define TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_CONFIG_H_

#include <vector>

#include "third_party/eigen3/Eigen/Core"

#include "tachyon/base/optional.h"
#include "tachyon/math/finite_fields/extension_field_traits_forward.h"
#include "tachyon/math/polynomials/univariate/evaluations_utils.h"

namespace tachyon::crypto {

template <typename MMCS>
struct TwoAdicFriConfig {
  uint32_t log_blowup;
  size_t num_queries;
  size_t proof_of_work_bits;
  MMCS mmcs;

  size_t Blowup() const { return size_t{1} << log_blowup; }
};

template <typename ExtF, typename Derived>
std::vector<ExtF> FoldMatrix(const ExtF& beta,
                             const Eigen::MatrixBase<Derived>& mat) {
  using F = typename math::ExtensionFieldTraits<ExtF>::BaseField;
  // We use the fact that
  //     pₑ(x²) = (p(x) + p(-x)) / 2
  //     pₒ(x²) = (p(x) - p(-x)) / (2x)
  // that is,
  //     pₑ(g²ⁱ) = (p(gⁱ) + p(gⁿᐟ²⁺ⁱ)) / 2
  //     pₒ(g²ⁱ) = (p(gⁱ) - p(gⁿᐟ²⁺ⁱ)) / (2gⁱ)
  // so
  //     result(g²ⁱ) = pₑ(g²ⁱ) + βpₒ(g²ⁱ)
  //                    = (1/2 + β/2gᵢₙᵥⁱ)p(gⁱ)
  //                    + (1/2 - β/2gᵢₙᵥⁱ)p(gⁿᐟ²⁺ⁱ)
  size_t rows = static_cast<size_t>(mat.rows());
  F w;
  CHECK(F::GetRootOfUnity(1 << (base::bits::CheckedLog2(rows) + 1), &w));
  ExtF w_inv = ExtF(unwrap(w.Inverse()));
  // TODO(ashjeong): implement a field function |TwoInv()| as the inverse of 2
  // is computed often.
  ExtF one_half = ExtF(unwrap(F(2).Inverse()));
  ExtF half_beta = beta * one_half;

  // β/2 times successive powers of gᵢₙᵥ
  std::vector<ExtF> powers =
      ExtF::GetBitRevIndexSuccessivePowers(rows, w_inv, half_beta);

  std::vector<ExtF> ret(rows);
  OMP_PARALLEL_FOR(size_t r = 0; r < rows; ++r) {
    const ExtF& lo = mat(r, 0);
    const ExtF& hi = mat(r, 1);
    ret[r] = (one_half + powers[r]) * lo + (one_half - powers[r]) * hi;
  }
  return ret;
}

// NOTE(ashjeong): |arity| is subject to change in the future
template <typename ExtF>
ExtF FoldRow(size_t index, size_t log_num_rows, const ExtF& beta,
             const std::vector<ExtF>& evals) {
  using F = typename math::ExtensionFieldTraits<ExtF>::BaseField;
  const size_t kArity = 2;
  const size_t kLogArity = 1;
  const ExtF& e0 = evals[0];
  const ExtF& e1 = evals[1];

  F w;
  CHECK(F::GetRootOfUnity(1 << (log_num_rows + kLogArity), &w));
  ExtF subgroup_start =
      ExtF(w.Pow(base::bits::ReverseBitsLen(index, log_num_rows)));

  CHECK(F::GetRootOfUnity(1 << kLogArity, &w));
  std::vector<ExtF> xs = ExtF::GetBitRevIndexSuccessivePowersSerial(
      kArity, ExtF(w), subgroup_start);
  // interpolate and evaluate at beta
  return e0 + unwrap((beta - xs[0]) * (e1 - e0) / (xs[1] - xs[0]));
}

}  // namespace tachyon::crypto

#endif  // TACHYON_CRYPTO_COMMITMENTS_FRI_TWO_ADIC_FRI_CONFIG_H_
