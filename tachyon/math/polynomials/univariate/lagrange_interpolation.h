// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_LAGRANGE_INTERPOLATION_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_LAGRANGE_INTERPOLATION_H_

#include <utility>
#include <vector>

#include "tachyon/base/memory/reusing_allocator.h"
#include "tachyon/base/parallelize.h"
#include "tachyon/base/template_util.h"
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

template <size_t MaxDegree, typename Container>
bool LagrangeInterpolate(
    const Container& points, const Container& evals,
    UnivariateDensePolynomial<base::container_value_t<Container>, MaxDegree>*
        ret) {
  using F = base::container_value_t<Container>;
  using Poly = UnivariateDensePolynomial<F, MaxDegree>;
  using Coeffs = UnivariateDenseCoefficients<F, MaxDegree>;

  if (points.size() != evals.size()) {
    LOG(ERROR) << "points and evals sizes don't match";
    return false;
  }

  if (points.size() == 1) {
    *ret = Poly(Coeffs({evals[0]}, true));
    return true;
  }

  // points = [x₀, x₁, ..., xₙ₋₁]
  // |denoms[i]| = Dᵢ = 1 / (xᵢ - x₀)(xᵢ - x₁)...(xᵢ - xₙ₋₁)
  std::vector<F> denoms(points.size(), F::One());
  base::Parallelize(denoms, [&points](absl::Span<F> chunk, size_t chunk_offset,
                                      size_t chunk_size) {
    size_t i = chunk_offset * chunk_size;
    for (F& denom : chunk) {
      for (size_t j = 0; j < points.size(); ++j) {
        if (i == j) continue;
        denom *= (points[i] - points[j]);
      }
      ++i;
    }
    CHECK(F::BatchInverseInPlaceSerial(chunk));
  });

  std::vector<std::vector<F, base::memory::ReusingAllocator<F>>> coeffs_sums =
      base::ParallelizeMap(evals, [&points, &denoms](absl::Span<const F> chunk,
                                                     size_t chunk_offset,
                                                     size_t chunk_size) {
        // See comments in |UnivariateDenseCoefficients::FromRoots()|.
        // clang-format off
        // NOTE(chokobole): This computes the polynomial whose roots are {x₀, ..., xᵢ₋₁, xᵢ₊₁, ..., xₙ₋₁}.
        // Nᵢ(X) = (X - x₀)...(X - xᵢ₋₁)(X - xᵢ₊₁)...(X - xₙ₋₁)
        //       = cₙ₋₁Xⁿ⁻¹ + cₙ₋₂Xⁿ⁻² + ... + c₁X¹ + c₀X⁰
        // |coeffs[i]| = cᵢ
        // clang-format on

        std::vector<F, base::memory::ReusingAllocator<F>> coeffs(points.size());
        std::vector<F, base::memory::ReusingAllocator<F>> coeffs_sum(
            points.size());

        size_t start = chunk_offset * chunk_size;
        for (size_t chunk_idx = 0; chunk_idx < chunk.size(); ++chunk_idx) {
          size_t i = start + chunk_idx;
          const F& eval = chunk[chunk_idx];
          const F& denom = denoms[i];

          coeffs[0] = F::One();
          for (size_t j = 1; j < points.size(); ++j) {
            coeffs[j] = F::Zero();
          }

          size_t k_start = 1;
          for (size_t j = 0; j < points.size(); ++j) {
            if (i == j) continue;
            for (size_t k = k_start; k > 0; --k) {
              coeffs[k] = coeffs[k - 1] - points[j] * coeffs[k];
            }
            ++k_start;
            coeffs[0] *= -points[j];
          }

          // P(X) = ∑ᵢ(P(Xᵢ) * Dᵢ * Nᵢ(X))
          //      = ∑ᵢ(P(Xᵢ) * Dᵢ * (cₙ₋₁Xⁿ⁻¹ + cₙ₋₂Xⁿ⁻² + ... + c₁X¹ + c₀X⁰))
          for (size_t j = 0; j < coeffs.size(); ++j) {
            coeffs[j] *= eval;
            coeffs[j] *= denom;
            coeffs_sum[j] += coeffs[j];
          }
        }
        return coeffs_sum;
      });
  for (size_t i = 1; i < coeffs_sums.size(); ++i) {
    for (size_t j = 0; j < points.size(); ++j) {
      coeffs_sums[0][j] += coeffs_sums[i][j];
    }
  }
  *ret = Poly(Coeffs(std::move(coeffs_sums[0]), true));
  return true;
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_LAGRANGE_INTERPOLATION_H_
