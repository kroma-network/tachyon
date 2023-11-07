// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_PROVER_QUERY_H_
#define TACHYON_ZK_BASE_PROVER_QUERY_H_

#include <utility>

#include "tachyon/zk/base/blinded_polynomial.h"

namespace tachyon::zk {

template <typename PCSTy>
class ProverQuery {
 public:
  using F = typename PCSTy::Field;
  using Poly = typename PCSTy::Poly;

  ProverQuery(const F& point, Ref<const BlindedPolynomial<Poly>> poly)
      : point_(point), poly_(poly) {}

  ProverQuery(F&& point, Ref<const BlindedPolynomial<Poly>> poly)
      : point_(std::move(point)), poly_(poly) {}

  const F& point() const { return point_; }
  Ref<const BlindedPolynomial<Poly>> poly() const { return poly_; }

  F Evaluate() const { return poly_->poly().Evaluate(*point_); }

 private:
  F point_;
  Ref<const BlindedPolynomial<Poly>> poly_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_PROVER_QUERY_H_
