// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_BLINDED_POLYNOMIAL_COMMITMENT_H_
#define TACHYON_ZK_BASE_BLINDED_POLYNOMIAL_COMMITMENT_H_

#include <utility>

#include "tachyon/zk/base/blinded_polynomial.h"

namespace tachyon::zk {

template <typename PCSTy>
class BlindedPolynomialCommitment {
 public:
  using F = typename PCSTy::Field;
  using Poly = typename PCSTy::Poly;
  using Commitment = typename PCSTy::Commitment;

  BlindedPolynomialCommitment() = default;
  BlindedPolynomialCommitment(const Poly& poly, const F& blind,
                              const Commitment& commitment)
      : poly_(poly), blind_(blind), commitment_(commitment) {}
  BlindedPolynomialCommitment(Poly&& poly, F&& blind, Commitment&& commitment)
      : poly_(std::move(poly)),
        blind_(std::move(blind)),
        commitment_(std::move(commitment)) {}

  const Poly& poly() const { return poly_; }
  const F& blind() const { return blind_; }
  const Commitment& commitment() const { return commitment_; }

  BlindedPolynomial<Poly> ToBlindedPolynomial() const& {
    return {poly_, blind_};
  }
  BlindedPolynomial<Poly> ToBlindedPolynomial() && {
    return {std::move(poly_), std::move(blind_)};
  }

 private:
  Poly poly_;
  F blind_;
  Commitment commitment_;
};

template <typename Domain, typename Evals, typename PCSTy,
          typename F = typename PCSTy::Field,
          typename Commitment = typename PCSTy::Commitment>
bool CommitEvalsWithBlind(const Domain* domain, const Evals& evals,
                          const PCSTy& pcs,
                          BlindedPolynomialCommitment<PCSTy>* out) {
  if (evals.NumElements() != domain->size()) return false;

  Commitment commitment;
  if (!pcs.CommitLagrange(evals, &commitment)) return false;

  // TODO(chokobole): Should sample blind from |Blinder|.
  *out = {domain->IFFT(evals), F::Random(), std::move(commitment)};
  return true;
}

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_BLINDED_POLYNOMIAL_COMMITMENT_H_
