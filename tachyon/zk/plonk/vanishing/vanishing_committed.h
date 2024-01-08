// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_COMMITTED_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_COMMITTED_H_

#include <utility>

#include "tachyon/zk/base/entities/entity_ty.h"

namespace tachyon::zk {

template <EntityTy EntityType, typename PCS>
class VanishingCommitted;

template <typename PCS>
class VanishingCommitted<EntityTy::kProver, PCS> {
 public:
  using F = typename PCS::Field;
  using Poly = typename PCS::Poly;

  VanishingCommitted() = default;
  VanishingCommitted(Poly&& random_poly, F&& random_blind)
      : random_poly_(std::move(random_poly)),
        random_blind_(std::move(random_blind)) {}

  const Poly& random_poly() { return random_poly_; }

  Poly&& TakeRandomPoly() && { return std::move(random_poly_); }
  F&& TakeRandomBlind() && { return std::move(random_blind_); }

 private:
  Poly random_poly_;
  F random_blind_;
};

template <typename PCS>
class VanishingCommitted<EntityTy::kVerifier, PCS> {
 public:
  using Commitment = typename PCS::Commitment;

  VanishingCommitted() = default;
  explicit VanishingCommitted(Commitment&& random_poly_commitment)
      : random_poly_commitment_(std::move(random_poly_commitment)) {}

  Commitment&& TakeRandomPolyCommitment() && {
    return std::move(random_poly_commitment_);
  }

 private:
  Commitment random_poly_commitment_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_COMMITTED_H_
