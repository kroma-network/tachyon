// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_EVALUATED_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_EVALUATED_H_

#include <utility>
#include <vector>

#include "tachyon/zk/base/entities/entity_ty.h"
#include "tachyon/zk/plonk/vanishing/vanishing_committed.h"

namespace tachyon::zk {

template <EntityTy EntityType, typename PCS>
class VanishingEvaluated;

template <typename PCS>
class VanishingEvaluated<EntityTy::kProver, PCS> {
 public:
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;

  VanishingEvaluated() = default;
  VanishingEvaluated(Poly&& h_poly, F&& h_blind,
                     VanishingCommitted<EntityTy::kProver, PCS>&& committed)
      : h_poly_(std::move(h_poly)),
        h_blind_(std::move(h_blind)),
        committed_(std::move(committed)) {}

  Poly&& TakeHPoly() && { return std::move(h_poly_); }
  F&& TakeHBlind() && { return std::move(h_blind_); }
  VanishingCommitted<EntityTy::kProver, PCS>&& TakeCommitted() && {
    return std::move(committed_);
  }

 private:
  Poly h_poly_;
  F h_blind_;
  VanishingCommitted<EntityTy::kProver, PCS> committed_;
};

template <typename PCS>
class VanishingEvaluated<EntityTy::kVerifier, PCS> {
 public:
  using F = typename PCS::Field;
  using Commitment = typename PCS::Commitment;

  VanishingEvaluated() = default;
  VanishingEvaluated(Commitment&& h_commitment,
                     Commitment&& random_poly_commitment, F&& expected_h_eval,
                     F&& random_eval)
      : h_commitment_(std::move(h_commitment)),
        random_poly_commitment_(std::move(random_poly_commitment)),
        expected_h_eval_(std::move(expected_h_eval)),
        random_eval_(std::move(random_eval)) {}

  const Commitment& h_commitment() const { return h_commitment_; }
  const Commitment& random_poly_commitment() const {
    return random_poly_commitment_;
  }

  F&& TakeExpectedHEval() && { return std::move(expected_h_eval_); }
  F&& TakeRandomEval() && { return std::move(random_eval_); }

 private:
  Commitment h_commitment_;
  Commitment random_poly_commitment_;
  F expected_h_eval_;
  F random_eval_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_EVALUATED_H_
