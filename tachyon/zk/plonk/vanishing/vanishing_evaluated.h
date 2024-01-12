// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_EVALUATED_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_EVALUATED_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/vanishing/vanishing_committed.h"

namespace tachyon::zk {

template <typename PCS>
class VanishingEvaluated {
 public:
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;

  VanishingEvaluated() = default;
  VanishingEvaluated(Poly&& h_poly, F&& h_blind,
                     VanishingCommitted<PCS>&& committed)
      : h_poly_(std::move(h_poly)),
        h_blind_(std::move(h_blind)),
        committed_(std::move(committed)) {}

  Poly&& TakeHPoly() && { return std::move(h_poly_); }
  F&& TakeHBlind() && { return std::move(h_blind_); }
  VanishingCommitted<PCS>&& TakeCommitted() && { return std::move(committed_); }

 private:
  Poly h_poly_;
  F h_blind_;
  VanishingCommitted<PCS> committed_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_EVALUATED_H_
