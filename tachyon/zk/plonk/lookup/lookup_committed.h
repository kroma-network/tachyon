// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_LOOKUP_LOOKUP_COMMITTED_H_
#define TACHYON_ZK_PLONK_LOOKUP_LOOKUP_COMMITTED_H_

#include <utility>

#include "tachyon/zk/base/prover.h"
#include "tachyon/zk/plonk/circuit/rotation.h"
#include "tachyon/zk/plonk/lookup/lookup_evaluated.h"

namespace tachyon::zk {

template <typename Poly>
class LookupCommitted {
 public:
  using F = typename Poly::Field;

  LookupCommitted() = default;
  LookupCommitted(BlindedPolynomial<Poly> permuted_input_poly,
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
  LookupEvaluated<Poly> Evaluate(Prover<PCSTy>* prover, const F& x) && {
    F x_inv = Rotation::Prev().RotateOmega(prover->domain(), x);
    F x_next = Rotation::Next().RotateOmega(prover->domain(), x);

    prover->Evaluate(product_poly_.poly(), x);
    prover->Evaluate(product_poly_.poly(), x_next);
    prover->Evaluate(permuted_input_poly_.poly(), x);
    prover->Evaluate(permuted_input_poly_.poly(), x_inv);
    prover->Evaluate(permuted_table_poly_.poly(), x);

    return {
        std::move(permuted_input_poly_),
        std::move(permuted_table_poly_),
        std::move(product_poly_),
    };
  }

 private:
  BlindedPolynomial<Poly> permuted_input_poly_;
  BlindedPolynomial<Poly> permuted_table_poly_;
  BlindedPolynomial<Poly> product_poly_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_LOOKUP_LOOKUP_COMMITTED_H_
