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
#include "tachyon/math/polynomials/univariate/univariate_polynomial.h"

namespace tachyon::math {

template <size_t MaxDegree, typename Container>
bool LagrangeInterpolate(
    const Container& points, const Container& evals,
    UnivariateDensePolynomial<typename Container::value_type, MaxDegree>* ret) {
  using F = typename Container::value_type;
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

  // points = [x₀, x₁, ..., xₙ]
  // denoms[i] = 1 / (xᵢ - x₀)(xᵢ - x₁)...(xᵢ - xₙ)
  std::vector<F> denoms = base::CreateVector(points.size(), F::One());
  base::Parallelize(denoms, [&points](absl::Span<F> chunk) {
    for (size_t i = 0; i < chunk.size(); ++i) {
      for (size_t j = 0; j < points.size(); ++j) {
        if (j != i) {
          chunk[i] *= (points[i] - points[j]);
        }
      }
    }
    F::BatchInverseInPlaceSerial(chunk);
  });

  // Final polynomial: ∑ᵢ evals[i] * numerators[i] * denoms[i]
  std::vector<F> final_coeffs = base::CreateVector(points.size(), F::Zero());
  base::Parallelize(final_coeffs, [&points, &evals,
                                   &denoms](absl::Span<F> chunk) {
    for (size_t i = 0; i < chunk.size(); ++i) {
      // Get numerators for each i:
      // numerators[i] = (x - x₀)...(x - xᵢ₋₁)(x - xᵢ₊₁)...(x - xₙ)
      std::vector<F> numerators = {F::One()};
      for (const F& x_j : points) {
        if (x_j != points[i]) {
          std::vector<F> product =
              base::CreateVector(numerators.size() + 1, F::Zero());
          for (size_t j = 0; j < numerators.size(); ++j) {
            product[j] -= x_j * numerators[j];
            product[j + 1] += numerators[j];
          }
          numerators = std::move(product);
        }
      }
      CHECK_EQ(numerators.size(), points.size());
      // clang-format off
      // evals[i] * (x - x₀)(x - x₁)...(x - xₙ) / (xᵢ - x₀)(xᵢ - x₁)...(xᵢ - xₙ)
      // clang-format on
      for (size_t j = 0; j < numerators.size(); ++j) {
        chunk[j] += evals[i] * numerators[j] * denoms[i];
      }
    }
  });
  *ret = Poly(Coeffs(std::move(final_coeffs)));
  return true;
}

}  // namespace tachyon::math

#endif  // TACHYON_MATH_POLYNOMIALS_UNIVARIATE_LAGRANGE_INTERPOLATION_H_
