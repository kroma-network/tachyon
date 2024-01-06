// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_VERIFIER_QUERY_H_
#define TACHYON_ZK_BASE_VERIFIER_QUERY_H_

#include <utility>

#include "tachyon/base/ref.h"

namespace tachyon::zk {

template <typename PCS>
class VerifierQuery {
 public:
  using F = typename PCS::Field;
  using Commitment = typename PCS::Commitment;

  VerifierQuery(const F& point, base::Ref<const Commitment> commitment,
                const F& evaluated)
      : point_(point), commitment_(commitment), evaluated_(evaluated) {}

  VerifierQuery(const F& point, base::Ref<const Commitment> commitment,
                F&& evaluated)
      : point_(point),
        commitment_(commitment),
        evaluated_(std::move(evaluated)) {}

  VerifierQuery(F&& point, base::Ref<const Commitment> commitment,
                F&& evaluated)
      : point_(std::move(point)),
        commitment_(commitment),
        evaluated_(std::move(evaluated)) {}

  const F& GetPoint() const { return point_; }

  base::Ref<const Commitment> GetCommitment() const { return commitment_; }

  const F& GetEval() const { return evaluated_; }

 private:
  F point_;
  base::Ref<const Commitment> commitment_;
  F evaluated_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_VERIFIER_QUERY_H_
