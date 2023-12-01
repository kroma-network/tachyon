// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_COMMITTED_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_COMMITTED_H_

#include <utility>
#include <vector>

#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/prover.h"
#include "tachyon/zk/plonk/permutation/permutation_evaluated.h"

namespace tachyon::zk {

// Committed polynomials for chunks of columns.
template <typename Poly>
class PermutationCommitted {
 public:
  explicit PermutationCommitted(
      std::vector<BlindedPolynomial<Poly>> product_polys)
      : product_polys_(std::move(product_polys)) {}

  const std::vector<BlindedPolynomial<Poly>>& product_polys() const {
    return product_polys_;
  }

  template <typename PCSTy, typename ExtendedDomain, typename F>
  PermutationEvaluated<Poly> Evaluate(Prover<PCSTy, ExtendedDomain>* prover,
                                      const F& x) && {
    int32_t blinding_factors =
        static_cast<int32_t>(prover->blinder().blinding_factors());

    for (size_t i = 0; i < product_polys_.size(); ++i) {
      const Poly& poly = product_polys_[i].poly();

      prover->Evaluate(poly, x);

      F x_next = Rotation::Next().RotateOmega(prover->domain(), x);
      prover->Evaluate(poly, x_next);

      // If we have any remaining sets to process, evaluate this set at ωᵘ
      // so we can constrain the last value of its running product to equal the
      // first value of the next set's running product, chaining them together.
      if (i != product_polys_.size() - 1) {
        F x_last =
            Rotation(-(blinding_factors + 1)).RotateOmega(prover->domain(), x);
        prover->Evaluate(poly, x_last);
      }
    }

    return PermutationEvaluated<Poly>(std::move(product_polys_));
  }

 private:
  std::vector<BlindedPolynomial<Poly>> product_polys_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_COMMITTED_H_
