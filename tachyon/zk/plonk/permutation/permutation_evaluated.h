// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_EVALUATED_H_
#define TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_EVALUATED_H_

#include <utility>
#include <vector>

#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/prover.h"
#include "tachyon/zk/base/prover_query.h"
#include "tachyon/zk/plonk/circuit/rotation.h"

namespace tachyon::zk {

template <typename Poly>
class PermutationEvaluated {
 public:
  PermutationEvaluated() = default;
  explicit PermutationEvaluated(
      std::vector<BlindedPolynomial<Poly>> product_polys)
      : product_polys_(std::move(product_polys)) {}

  const std::vector<BlindedPolynomial<Poly>>& product_polys() const {
    return product_polys_;
  }

  template <typename PCSTy, typename F>
  std::vector<ProverQuery<PCSTy>> Open(const Prover<PCSTy>* prover,
                                       const F& x) const {
    std::vector<ProverQuery<PCSTy>> ret;
    ret.reserve(product_polys_.size() * 3 - 1);

    F x_next = Rotation::Next().RotateOmega(prover->domain(), x);
    for (const BlindedPolynomial<Poly>& blinded_poly : product_polys_) {
      ret.emplace_back(x, blinded_poly.ToRef());
      ret.emplace_back(x_next, blinded_poly.ToRef());
    }

    int32_t blinding_factors =
        static_cast<int32_t>(prover->blinder().blinding_factors());
    F x_last =
        Rotation(-(blinding_factors + 1)).RotateOmega(prover->domain(), x);
    for (auto it = product_polys_.rbegin() + 1; it != product_polys_.rend();
         ++it) {
      ret.emplace_back(x_last, it->ToRef());
    }
    return ret;
  }

 private:
  std::vector<BlindedPolynomial<Poly>> product_polys_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_PERMUTATION_PERMUTATION_EVALUATED_H_
