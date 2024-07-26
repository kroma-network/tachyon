// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_UTILS_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_UTILS_H_

#include <memory_resource>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/parallelize.h"
#include "tachyon/base/types/always_false.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/entities/prover_base.h"
#include "tachyon/zk/plonk/constraint_system/gate.h"
#include "tachyon/zk/plonk/halo2/config.h"

namespace tachyon::zk::plonk {

// Calculate ζ = g^((2ˢ * T) / 3).
template <typename F>
F GetZeta() {
  using BigInt = typename F::BigIntTy;
  CHECK_EQ(F::Config::kTrace % BigInt(3), BigInt(0));
  BigInt exp = F::Config::kTrace;
  // NOTE(chokobole): The result of the exponential operation does not exceed
  // the modulus of the scalar field.
  exp.MulBy2ExpInPlace(F::Config::kTwoAdicity);
  CHECK(exp /= BigInt(3));
  return F::FromMontgomery(F::Config::kSubgroupGenerator).Pow(exp);
}

// NOTE(TomTaehoonKim): The returning value of |GetZeta()| is different from the
// one in
// https://github.com/privacy-scaling-explorations/halo2curves/blob/v0.6.1/src/bn256/fr.rs#L145-L151.
// This is an ugly way to produce a same result with Halo2Curves but we will
// remove once we don't have to match it against Halo2 any longer in the
// future.
// Calculate ζ = g^((2ˢ * T) / 3)².
template <typename F>
F GetHalo2Zeta() {
  static F cache = F::Zero();
  if (cache.IsZero()) {
    halo2::Config& config = halo2::GetConfig();
    if constexpr (std::is_same_v<F, math::bn254::Fr>) {
      if (config.vendor == halo2::Vendor::kScroll) {
        // See
        // https://github.com/scroll-tech/halo2curves/blob/3c268b4/src/bn256/fr.rs#L108-L114.
        cache = GetZeta<F>();
      } else {
        cache = GetZeta<F>().Square();
      }
    } else {
      static_assert(base::AlwaysFalse<F>);
    }
  }
  return cache;
}

// This divides the polynomial (in the extended domain) by the vanishing
// polynomial of the 2ᵏ size domain.
template <typename Domain, typename ExtendedDomain, typename ExtendedEvals>
ExtendedEvals& DivideByVanishingPolyInPlace(
    ExtendedEvals& evals, const ExtendedDomain* extended_domain,
    const Domain* domain) {
  using F = typename ExtendedEvals::Field;

  CHECK_EQ(evals.NumElements(), extended_domain->size());

  const F zeta = GetHalo2Zeta<F>();

  // Compute the evaluations of t(X) = Xⁿ - 1 in the coset evaluation domain.
  // We don't have to compute all of them, because it will repeat.
  const size_t kTEvaluationsSize = size_t{1}
                                   << (extended_domain->log_size_of_group() -
                                       domain->log_size_of_group());
  // |coset_gen_pow_n| = w'ⁿ where w' is generator of extended domain.
  const F coset_gen_pow_n = extended_domain->group_gen().Pow(domain->size());
  const F zeta_pow_n = zeta.Pow(domain->size());
  std::vector<F> t_evaluations(kTEvaluationsSize);
  // |t_evaluations| = [ζⁿ - 1, ζⁿ * w'ⁿ - 1, ζⁿ * w'²ⁿ - 1, ...]
  base::Parallelize(t_evaluations, [&coset_gen_pow_n, &zeta_pow_n](
                                       absl::Span<F> chunk, size_t chunk_offset,
                                       size_t chunk_size) {
    size_t start = chunk_offset * chunk_size;
    F zeta_i = zeta_pow_n * coset_gen_pow_n.Pow(start);
    for (F& coeff : chunk) {
      coeff = zeta_i - F::One();
      zeta_i *= coset_gen_pow_n;
    }
    CHECK(F::BatchInverseInPlaceSerial(chunk));
  });

  // Multiply the inverse to obtain the quotient polynomial in the coset
  // evaluation domain.
  std::pmr::vector<F>& evaluations = evals.evaluations();
  OPENMP_PARALLEL_FOR(size_t i = 0; i < evaluations.size(); ++i) {
    evaluations[i] *= t_evaluations[i % t_evaluations.size()];
  }

  return evals;
}

// Given a |poly| of coefficients  [a₀, a₁, a₂, ...], this returns
// [a₀, ζa₁, ζ²a₂, a₃, ζa₄, ζ²a₅, a₆, ...], where ζ is a cube root of unity in
// the multiplicative subgroup with order (p - 1), i.e. ζ³ = 1.
//
// |into_coset| should be set to true when moving into the coset, and false
// when moving out. This toggles the choice of ζ.
template <typename Poly>
void DistributePowersZeta(Poly& poly, bool into_coset) {
  using F = typename Poly::Field;

  F zeta = GetHalo2Zeta<F>();
  F zeta_inv = zeta.Square();
  F coset_powers[] = {into_coset ? zeta : zeta_inv,
                      into_coset ? zeta_inv : zeta};

  std::pmr::vector<F>& coeffs = poly.coefficients().coefficients();
  OPENMP_PARALLEL_FOR(size_t i = 0; i < coeffs.size(); ++i) {
    size_t j = i % 3;
    if (j == 0) continue;
    coeffs[i] *= coset_powers[j - 1];
  }
}

// This takes us from an n-length coefficient vector into a coset of the
// extended evaluation domain.
template <typename Poly, typename ExtendedDomain,
          typename ExtendedEvals = typename ExtendedDomain::Evals>
ExtendedEvals CoeffToExtended(Poly&& poly,
                              const ExtendedDomain* extended_domain) {
  using ExtendedPoly = typename ExtendedDomain::DensePoly;
  using Coefficients = typename ExtendedPoly::Coefficients;

  DistributePowersZeta(poly, true);

  ExtendedPoly extended_poly(
      Coefficients(std::move(poly).TakeCoefficients().TakeCoefficients()));
  return extended_domain->FFT(std::move(extended_poly));
}

// This takes us from the extended evaluation domain and gets us the quotient
// polynomial coefficients.
//
// This function will crash if the provided vector is not the correct length.
template <typename ExtendedEvals, typename ExtendedDomain,
          typename ExtendedPoly = typename ExtendedDomain::DensePoly>
ExtendedPoly ExtendedToCoeff(ExtendedEvals&& evals,
                             const ExtendedDomain* extended_domain) {
  CHECK_EQ(evals.NumElements(), extended_domain->size());

  ExtendedPoly poly = extended_domain->IFFT(std::move(evals));

  // Distribute powers to move from coset; opposite from the
  // transformation we performed earlier.
  DistributePowersZeta(poly, false);

  return poly;
}

template <typename F>
std::pmr::vector<F> BuildExtendedColumnWithColumns(
    const std::vector<std::vector<F>>& columns) {
  CHECK(!columns.empty());
  size_t cols = columns.size();
  size_t rows = columns[0].size();

  std::pmr::vector<F> flattened_transposed_columns(cols * rows);
  OPENMP_PARALLEL_NESTED_FOR(size_t i = 0; i < columns.size(); ++i) {
    for (size_t j = 0; j < rows; ++j) {
      flattened_transposed_columns[j * cols + i] = columns[i][j];
    }
  }
  return flattened_transposed_columns;
}

template <typename F>
static size_t GetNumVanishingEvals(size_t num_circuits,
                                   const std::vector<Gate<F>>& gates) {
  size_t num_polys = std::accumulate(gates.begin(), gates.end(), 0,
                                     [](size_t acc, const Gate<F>& gate) {
                                       return acc + gate.polys().size();
                                     });
  return num_circuits * num_polys;
}

template <typename PCS>
constexpr static size_t GetNumVanishingOpenings(size_t num_circuits,
                                                size_t num_advice_queries,
                                                size_t num_instance_queries,
                                                size_t num_fixed_queries) {
  size_t ret = num_advice_queries;
  if (PCS::kQueryInstance) {
    ret += num_instance_queries;
  }
  ret *= num_circuits;
  ret += num_fixed_queries;
  ret += 2;
  return ret;
}

}  // namespace tachyon::zk::plonk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_UTILS_H_
