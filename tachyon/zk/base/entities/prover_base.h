// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_BASE_ENTITIES_PROVER_BASE_H_
#define TACHYON_ZK_BASE_ENTITIES_PROVER_BASE_H_

#include <memory>
#include <utility>

#include "tachyon/base/logging.h"
#include "tachyon/zk/base/blinded_polynomial.h"
#include "tachyon/zk/base/blinder.h"
#include "tachyon/zk/base/entities/entity.h"

namespace tachyon::zk {

template <typename PCSTy>
class ProverBase : public Entity<PCSTy> {
 public:
  using F = typename PCSTy::Field;
  using Evals = typename PCSTy::Evals;
  using Poly = typename PCSTy::Poly;
  using Commitment = typename PCSTy::Commitment;

  ProverBase(PCSTy&& pcs,
             std::unique_ptr<crypto::TranscriptWriter<Commitment>> writer,
             Blinder<PCSTy>&& blinder)
      : Entity<PCSTy>(std::move(pcs), std::move(writer)),
        blinder_(std::move(blinder)) {}

  Blinder<PCSTy>& blinder() { return blinder_; }

  crypto::TranscriptWriter<Commitment>* GetWriter() {
    return this->transcript()->ToWriter();
  }

  size_t GetUsableRows() const {
    return this->domain_->size() - (blinder_.blinding_factors() + 1);
  }

  [[nodiscard]] bool Commit(const Poly& poly) {
    Commitment commitment;
    if (!this->pcs_.Commit(poly, &commitment)) return false;
    return GetWriter()->WriteToProof(commitment);
  }

  template <typename ContainerTy>
  [[nodiscard]] bool Commit(const ContainerTy& coeffs) {
    Commitment commitment;
    if (!this->pcs_.DoCommit(coeffs, &commitment)) return false;
    return GetWriter()->WriteToProof(commitment);
  }

  [[nodiscard]] bool CommitEvals(const Evals& evals) {
    if (evals.NumElements() != this->domain_->size()) return false;

    Commitment commitment;
    if (!this->pcs_.CommitLagrange(evals, &commitment)) return false;
    return GetWriter()->WriteToProof(commitment);
  }

  [[nodiscard]] bool CommitEvalsWithBlind(const Evals& evals,
                                          BlindedPolynomial<Poly>* out) {
    if (!CommitEvals(evals)) return false;
    *out = {this->domain_->IFFT(evals), blinder_.Generate()};
    return true;
  }

  [[nodiscard]] bool Evaluate(const Poly& poly, const F& x) {
    F result = poly.Evaluate(x);
    return GetWriter()->WriteToProof(result);
  }

 protected:
  Blinder<PCSTy> blinder_;
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_BASE_ENTITIES_PROVER_BASE_H_
