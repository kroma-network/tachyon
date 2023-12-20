// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_UTILS_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_UTILS_H_

#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/parallelize.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/entities/prover_base.h"

namespace tachyon::zk {

// Calculate ζ = g^((2ˢ * T) / 3).
template <typename F>
constexpr F GetZeta() {
  CHECK_EQ(F::Config::kTrace % math::BigInt<F::kLimbNums>(3),
           math::BigInt<F::kLimbNums>(0));
  return F::FromMontgomery(F::Config::kSubgroupGenerator)
      .Pow(F(2).Pow(F::Config::kTwoAdicity).ToBigInt() * F::Config::kTrace /
           math::BigInt<F::kLimbNums>(3));
}

// NOTE(TomTaehoonKim): The returning value of |GetZeta()| is different from the
// one in
// https://github.com/kroma-network/halo2curves/blob/c0ac1935e5da2a620204b5b011be2c924b1e0155/src/bn256/fr.rs#L112-L118.
// This is an ugly way to produce a same result with Halo2Curves but we will
// remove once we don't have to match it against Halo2 any longer in the
// future.
// Calculate ζ = g^((2ˢ * T) / 3)².
template <typename F>
constexpr F GetHalo2Zeta() {
  return GetZeta<F>().Square();
}

// This divides the polynomial (in the extended domain) by the vanishing
// polynomial of the 2ᵏ size domain.
template <typename F, typename Domain, typename ExtendedDomain,
          typename ExtendedEvals>
ExtendedEvals& DivideByVanishingPolyInPlace(
    ExtendedEvals& evals, const ExtendedDomain* extended_domain,
    const Domain* domain) {
  CHECK_EQ(evals.NumElements(), extended_domain->size());

  const F zeta = GetHalo2Zeta<F>();

  // Compute the evaluations of t(X) = Xⁿ - 1 in the coset evaluation domain.
  // We don't have to compute all of them, because it will repeat.
  const size_t t_evaluations_size = size_t{1}
                                    << (extended_domain->log_size_of_group() -
                                        domain->log_size_of_group());
  const F coset_gen_pow_n = extended_domain->group_gen().Pow(domain->size());
  const std::vector<F> powers_of_coset_gen =
      F::GetSuccessivePowers(t_evaluations_size, coset_gen_pow_n);

  std::vector<F> t_evaluations;
  t_evaluations.resize(t_evaluations_size);

  const F zeta_pow_n = zeta.Pow(domain->size());
  CHECK(F::MultiScalarMul(zeta_pow_n * coset_gen_pow_n, powers_of_coset_gen,
                          &t_evaluations));
  CHECK_EQ(t_evaluations.size(),
           size_t{1} << (extended_domain->log_size_of_group() -
                         domain->log_size_of_group()));

  // Subtract 1 from each to give us t_evaluations[i] = t(zeta *
  // extended_omegaⁱ)
  // TODO(TomTaehoonKim): Consider implementing "translate" function.
  base::Parallelize(t_evaluations, [](absl::Span<F> chunk) {
    for (F& coeff : chunk) {
      coeff -= F::One();
    }
  });

  F::BatchInverseInPlace(t_evaluations);

  // Multiply the inverse to obtain the quotient polynomial in the coset
  // evaluation domain.
  std::vector<F>& evaluations = evals.evaluations();
  base::Parallelize(evaluations,
                    [&t_evaluations](absl::Span<F> chunk, size_t chunk_idx,
                                     size_t chunk_size) {
                      size_t index = chunk_idx * chunk_size;
                      for (F& h : chunk) {
                        h *= t_evaluations[index % t_evaluations.size()];
                        ++index;
                      }
                    });

  return evals;
}

// This divides the polynomial (in the extended domain) by the vanishing
// polynomial of the 2ᵏ size domain.
template <typename F, typename Domain, typename ExtendedDomain,
          typename ExtendedEvals>
ExtendedEvals DivideByVanishingPoly(const ExtendedEvals& evals,
                                    const ExtendedDomain* extended_domain,
                                    const Domain* domain) {
  ExtendedEvals ret = evals;
  DivideByVanishingPolyInPlace<F>(ret, extended_domain, domain);
  return ret;
}

// Given a |poly| of coefficients  [a₀, a₁, a₂, ...], this returns
// [a₀, ζa₁, ζ²a₂, a₃, ζa₄, ζ²a₅, a₆, ...], where ζ is a cube root of unity in
// the multiplicative subgroup with order (p - 1), i.e. ζ³ = 1.
//
// |into_coset| should be set to true when moving into the coset, and false
// when moving out. This toggles the choice of ζ.
template <typename F, typename ExtendedPoly>
void DistributePowersZeta(ExtendedPoly& poly, bool into_coset) {
  F zeta = GetHalo2Zeta<F>();
  F zeta_inv = zeta.Square();
  std::vector<F> coset_powers{into_coset ? zeta : zeta_inv,
                              into_coset ? zeta_inv : zeta};

  std::vector<F> coeffs = poly.coefficients().coefficients();
  base::Parallelize(coeffs,
                    [&coset_powers](absl::Span<F> chunk, size_t chunk_idx,
                                    size_t chunk_size) {
                      size_t i = chunk_idx * chunk_size;
                      for (F& a : chunk) {
                        // Distribute powers to move into/from coset
                        size_t j = i % (coset_powers.size() + 1);
                        if (j != 0) {
                          a *= coset_powers[j - 1];
                        }
                        ++i;
                      }
                    });
}

// This takes us from the extended evaluation domain and gets us the quotient
// polynomial coefficients.
//
// This function will panic if the provided vector is not the correct length.
template <typename F, typename ExtendedPoly, typename ExtendedEvals,
          typename ExtendedDomain>
ExtendedPoly ExtendedToCoeff(const ExtendedEvals& evals,
                             const ExtendedDomain* extended_domain) {
  CHECK_EQ(evals.NumElements(), extended_domain->size());

  ExtendedPoly poly = extended_domain->IFFT(evals);

  // Distribute powers to move from coset; opposite from the
  // transformation we performed earlier.
  DistributePowersZeta<F>(poly, false);

  return poly;
}

template <typename Domain, typename Poly, typename F,
          typename Evals = typename Domain::Evals>
Evals CoeffToExtendedPart(const Domain* domain,
                          const BlindedPolynomial<Poly>& poly, const F& zeta,
                          const F& extended_omega_factor) {
  Poly cloned = poly.poly();
  Domain::DistributePowers(cloned, zeta * extended_omega_factor);
  return domain->FFT(cloned);
}

template <typename Domain, typename Poly, typename F,
          typename Evals = typename Domain::Evals>
Evals CoeffToExtendedPart(const Domain* domain, const Poly& poly, const F& zeta,
                          const F& extended_omega_factor) {
  Poly cloned = poly;
  Domain::DistributePowers(cloned, zeta * extended_omega_factor);
  return domain->FFT(cloned);
}

template <typename Domain, typename Poly, typename F,
          typename Evals = typename Domain::Evals>
std::vector<Evals> CoeffsToExtendedPart(const Domain* domain,
                                        absl::Span<Poly> polys, const F& zeta,
                                        const F& extended_omega_factor) {
  return base::Map(
      polys, [domain, &zeta, &extended_omega_factor](const Poly& poly) {
        return CoeffToExtendedPart(domain, poly, zeta, extended_omega_factor);
      });
}

template <typename F>
std::vector<F> BuildExtendedColumnWithColumns(
    std::vector<std::vector<F>>&& columns) {
  CHECK(!columns.empty());
  size_t cols = columns.size();
  size_t rows = columns[0].size();

  std::vector<std::vector<F>> transposed = base::CreateVector(
      rows, [cols]() { return base::CreateVector(cols, F::Zero()); });
  for (size_t i = 0; i < columns.size(); ++i) {
    base::Parallelize(transposed, [i, &src_column = columns[i]](
                                      absl::Span<std::vector<F>> dst_columns,
                                      size_t chunk_idx, size_t chunk_size) {
      size_t start = chunk_idx * chunk_size;
      for (size_t j = 0; j < dst_columns.size(); ++j) {
        dst_columns[j][i] = src_column[start + j];
      }
    });
  }
  std::vector<F> flattened_columns;
  flattened_columns.reserve(cols * rows);
  for (std::vector<F>& column : transposed) {
    flattened_columns.insert(flattened_columns.end(),
                             std::make_move_iterator(column.begin()),
                             std::make_move_iterator(column.end()));
  }
  return flattened_columns;
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_UTILS_H_
