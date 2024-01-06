// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_VANISHING_VANISHING_PARTIALLY_EVALUATED_H_
#define TACHYON_ZK_PLONK_VANISHING_VANISHING_PARTIALLY_EVALUATED_H_

#include <utility>
#include <vector>

namespace tachyon::zk {

template <typename PCS>
class VanishingPartiallyEvaluated {
 public:
  using F = typename PCS::Field;
  using Commitment = typename PCS::Commitment;

  VanishingPartiallyEvaluated() = default;
  VanishingPartiallyEvaluated(std::vector<Commitment>&& h_commitments,
                              Commitment&& random_poly_commitment,
                              F&& random_eval)
      : h_commitments_(std::move(h_commitments)),
        random_poly_commitment_(std::move(random_poly_commitment)),
        random_eval_(std::move(random_eval)) {}

  const std::vector<Commitment>& h_commitments() const {
    return h_commitments_;
  }

  Commitment&& TakeRandomPolyCommitment() && {
    return std::move(random_poly_commitment_);
  }
  F&& TakeRandomEval() && { return std::move(random_eval_); }

 private:
  std::vector<Commitment> h_commitments_;
  Commitment random_poly_commitment_;
  F random_eval_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_VANISHING_VANISHING_PARTIALLY_EVALUATED_H_
