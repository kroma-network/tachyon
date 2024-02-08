// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_MATH_POLYNOMIALS_UNIVARIATE_LAGRANGE_INTERPOLATION_H_
#define TACHYON_MATH_POLYNOMIALS_UNIVARIATE_LAGRANGE_INTERPOLATION_H_

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
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
    *ret = Poly(Coeffs({evals[0]}));
    return true;
  }

  // points = [x₀, x₁, ..., xₙ₋₁]
  // |denoms[i]| = Dᵢ = 1 / (xᵢ - x₀)(xᵢ - x₁)...(xᵢ - xₙ₋₁)
  std::vector<F> denoms = base::CreateVector(points.size(), F::One());
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

  std::vector<F> final_coeffs = base::CreateVector(points.size(), F::Zero());
  for (size_t i = 0; i < points.size(); ++i) {
    // See comments in |UnivariateDenseCoefficients::FromRoots()|.
    // clang-format off
    // NOTE(chokobole): This computes the polynomial whose roots are {x₀, ..., xᵢ₋₁, xᵢ₊₁, ..., xₙ₋₁}.
    // Nᵢ(X) = (X - x₀)...(X - xᵢ₋₁)(X - xᵢ₊₁)...(X - xₙ₋₁)
    //       = cₙ₋₁Xⁿ⁻¹ + cₙ₋₂Xⁿ⁻² + ... + c₁X¹ + c₀X⁰
    // |coeffs[i]| = cᵢ
    // clang-format on
    std::vector<F> coeffs = base::CreateVector(points.size(), F::Zero());
    coeffs[0] = F::One();
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
    OPENMP_PARALLEL_FOR(size_t j = 0; j < coeffs.size(); ++j) {
      coeffs[j] *= evals[i];
      coeffs[j] *= denoms[i];
      final_coeffs[j] += coeffs[j];
    }
  }
  *ret = Poly(Coeffs(std::move(final_coeffs)));
  return true;
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_LAGRANGE_INTERPOLATION_H_
