// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LOOKUP_LOOKUP_EVALUATED_H_
#define TACHYON_ZK_PLONK_LOOKUP_LOOKUP_EVALUATED_H_

#include <utility>
#include <vector>

#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/prover.h"
#include "tachyon/zk/base/prover_query.h"

namespace tachyon::zk {

template <typename Poly>
class LookupEvaluated {
 public:
  using F = typename Poly::Field;

  LookupEvaluated() = default;
  LookupEvaluated(BlindedPolynomial<Poly> permuted_input_poly,
                  BlindedPolynomial<Poly> permuted_table_poly,
                  BlindedPolynomial<Poly> product_poly)
      : permuted_input_poly_(std::move(permuted_input_poly)),
        permuted_table_poly_(std::move(permuted_table_poly)),
        product_poly_(std::move(product_poly)) {}

  const BlindedPolynomial<Poly>& permuted_input_poly() const {
    return permuted_input_poly_;
  }
  const BlindedPolynomial<Poly>& permuted_table_poly() const {
    return permuted_table_poly_;
  }
  const BlindedPolynomial<Poly>& product_poly() const { return product_poly_; }

  template <typename PCSTy>
  std::vector<ProverQuery<PCSTy>> Open(Prover<PCSTy>* prover,
                                       const F& x) const {
    F x_inv = Rotation::Prev().RotateOmega(prover->domain(), x);
    F x_next = Rotation::Next().RotateOmega(prover->domain(), x);

    return {ProverQuery<PCSTy>(x, product_poly_.ToRef()),
            ProverQuery<PCSTy>(x, permuted_input_poly_.ToRef()),
            ProverQuery<PCSTy>(std::move(x), permuted_table_poly_.ToRef()),
            ProverQuery<PCSTy>(std::move(x_inv), permuted_input_poly_.ToRef()),
            ProverQuery<PCSTy>(std::move(x_next), product_poly_.ToRef())};
  }

 private:
  BlindedPolynomial<Poly> permuted_input_poly_;
  BlindedPolynomial<Poly> permuted_table_poly_;
  BlindedPolynomial<Poly> product_poly_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LOOKUP_LOOKUP_EVALUATED_H_
